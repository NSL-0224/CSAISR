"""CSA-ISR: KVCA with nested dual-condition rejection (DCR).

Retains upstream CrossScaleAttention projections, patch extraction, dynamic
convolution / transposed convolution, downsampling, cropping and /6 scaling.
Borrowed ideas: SAT-style density/separation center selection; proportional
attention's log-multiplicity correction. These are NOT claimed as new.
Main method: joint K/V/spatial grouping, DCR, paired similarity-weighted means, and unified
matching/reconstruction. Optional Gaussian spatial reading is disabled by
default and is not part of the main embodiment. No PSNR claim is made.
"""
import math
from contextlib import nullcontext

import torch
from torch import nn
from torch.nn import functional as F

from mmedited.models.common.arch_csnln import (
    CrossScaleAttention, extract_image_patches, same_padding)


class PatchKVAttention(CrossScaleAttention):
    def __init__(self, channel=64, scale=(2,), keep_ratio=0.5, geometry=False,
                 reject_thresholds=(0.5, 0.5), aggregation_temperature=1.0):
        scales = list(scale)
        if not scales or any(s not in (2, 3, 4) for s in scales):
            raise ValueError('PatchKVAttention supports original scales 2, 3, 4')
        if not math.isfinite(keep_ratio) or not 0 < keep_ratio <= 1:
            raise ValueError('keep_ratio must be finite and in (0, 1]')
        if not math.isfinite(aggregation_temperature) or aggregation_temperature <= 0:
            raise ValueError('aggregation_temperature must be finite and positive')
        if reject_thresholds is not None:
            if len(reject_thresholds) != 2 or any(not math.isfinite(t) or t < 0 for t in reject_thresholds):
                raise ValueError('reject_thresholds must be two finite nonnegative distances or None')
            reject_thresholds = tuple(float(t) for t in reject_thresholds)
        super().__init__(channel=channel, scale=scales)
        self.keep_ratio = keep_ratio
        self.aggregation_temperature = float(aggregation_temperature)
        self.geometry = bool(geometry)
        self.region_size = 4
        # None reproduces v1 fixed-budget merging. With thresholds, reject
        # if either K or V differs too much from the assigned ORIGINAL center.
        self.reject_thresholds = reject_thresholds
        # Diagnostics from latest forward, not checkpoint state. Inspect on
        # the rank/device of interest; not automatically reduced across DDP.
        self.last_bank_stats = []
        if self.geometry:
            # softplus(-6) ~= .00248; weak but not zero spatial prior.
            self.geometry_log_strength = nn.Parameter(torch.tensor(-6.0))

    @staticmethod
    def _positions(h, w, scale, device):
        # Actual center of the original (3*s)x(3*s), stride-s source patches,
        # measured in padded LR pixel-index coordinates. Not HR query positions.
        yy = (torch.arange(h, device=device, dtype=torch.float32) + .5)*scale - .5
        xx = (torch.arange(w, device=device, dtype=torch.float32) + .5)*scale - .5
        return torch.stack((yy[:, None].expand(h, w), xx[None, :].expand(h, w)), -1)

    def _merge_region(self, key, value, position, scale):
        """Hard grouping/DCR, then SAT-style cosine-weighted paired means.

        Similarity is measured on K against its original assigned center.
        This adapts SAT's averaging rule, not its full token pipeline or FNR.
        """
        b, n, _ = key.shape
        count = max(1, math.ceil(n*self.keep_ratio))
        if count == n:
            return key, value, position, torch.zeros_like(position), key.new_ones(b, n)
        with torch.no_grad():
            kd, vd = key.detach(), value.detach()
            kn = kd.square().sum(-1)
            vn = vd.square().sum(-1)
            dk = (kn[:, :, None]+kn[:, None, :]-2*(kd @ kd.transpose(1, 2))).clamp_min(0)
            dv = (vn[:, :, None]+vn[:, None, :]-2*(vd @ vd.transpose(1, 2))).clamp_min(0)
            dv = dv / (.5*(vn[:, :, None]+vn[:, None, :])+1e-8)
            # Nominal region side in LR units, including partial edge regions.
            diff = position[:, :, None]-position[:, None, :]
            side = float(self.region_size*scale)
            dp = diff.square().sum(-1)/(side*side)
            distance = dk+dv+dp
            eye = torch.eye(n, device=key.device, dtype=torch.bool)[None]
            neighbors = distance.masked_fill(eye, float('inf')).topk(
                min(5, n-1), dim=-1, largest=False).values
            density = (-neighbors.mean(-1)).exp()
            order = torch.arange(n, device=key.device)
            higher = density[:, None, :] > density[:, :, None]
            ties = density[:, None, :] == density[:, :, None]
            higher = higher | (ties & (order[None, :] < order[:, None])[None])
            delta = distance.masked_fill(~higher, float('inf')).amin(-1)
            delta = torch.where(higher.any(-1), delta, distance.amax(-1))
            # Stable order breaks equal center scores by the original index.
            centers = torch.argsort(density*delta, dim=-1, descending=True,
                                    stable=True)[:, :count]
            to_centers = distance.gather(2, centers[:, None].expand(b, n, count))
            assignment = to_centers.argmin(-1)
            assignment.scatter_(1, centers, torch.arange(count, device=key.device)[None].expand(b, -1))
            chosen = centers.gather(1, assignment)
            if self.reject_thresholds is not None:
                chosen_k = kd.gather(1, chosen[..., None].expand_as(kd))
                chosen_v = vd.gather(1, chosen[..., None].expand_as(vd))
                # Direct differences avoid Gram-matrix cancellation at zero
                # thresholds. Spatial preference selects the center, but is
                # deliberately NOT a condition in the rejection test.
                error_k = (kd-chosen_k).square().sum(-1)
                error_v = (vd-chosen_v).square().sum(-1) / (
                    .5*(vd.square().sum(-1)+chosen_v.square().sum(-1))+1e-8)
                reject = (error_k > self.reject_thresholds[0]) | (error_v > self.reject_thresholds[1])
                reject.scatter_(1, centers, False)
                # Each rejected candidate gets a fresh singleton, without
                # absorbing other candidates or triggering another assignment.
                singleton = count+reject.long().cumsum(-1)-1
                assignment = torch.where(reject, singleton, assignment)
                # A rejected candidate becomes its own reference center.
                chosen = torch.where(reject, order[None].expand(b, -1), chosen)
                count += int(reject.sum(-1).max().item())

        def summed(x):
            result = x.new_zeros(b, count, x.shape[-1])
            return result.scatter_add(1, assignment[..., None].expand_as(x), x)

        mass = summed(key.new_ones(b, n, 1))
        # Different images may have different numbers of singleton groups.
        # Zero-mass slots pad the batch only and must never enter attention.
        center_key = key.gather(1, chosen[..., None].expand_as(key))
        similarity = F.cosine_similarity(key, center_key, dim=-1, eps=1e-8)
        # Subtract each group's maximum before exp for numerical stability.
        # Regions contain at most 16 candidates; padded groups are empty.
        membership = assignment[:, None, :] == torch.arange(count, device=key.device)[None, :, None]
        group_max = similarity.detach()[:, None, :].expand(b, count, n).masked_fill(
            ~membership, float('-inf')).amax(-1)
        shifted = similarity - group_max.gather(1, assignment)
        weights = (shifted / self.aggregation_temperature).exp().unsqueeze(-1)
        denominator = summed(weights).clamp_min(1e-8)
        key_mean = summed(key * weights)/denominator
        value_mean = summed(value * weights)/denominator
        mean_position = summed(position * weights)/denominator
        centered = position-mean_position.gather(1, assignment[..., None].expand(b, n, 2))
        variance = summed(centered.square() * weights)/denominator
        # No FNR or re-normalization. K/V share weights. Mass remains the
        # integer member count for log-multiplicity correction, not weight sum.
        # For unequal keys that correction is approximate, not exact attention.
        return key_mean, value_mean, mean_position, variance, mass.squeeze(-1)

    def _aggregate_bank(self, key, value, h, w, scale):
        b, n, _ = key.shape
        position = self._positions(h, w, scale, key.device).reshape(1, n, 2).expand(b, -1, -1)
        if self.keep_ratio == 1:
            # Preserve original candidate order for the identity control.
            return key, value, position, torch.zeros_like(position), key.new_ones(b, n)
        index_map = torch.arange(n, device=key.device).reshape(h, w)
        groups = []
        for top in range(0, h, self.region_size):
            for left in range(0, w, self.region_size):
                indices = index_map[top:top+self.region_size, left:left+self.region_size].reshape(-1)
                groups.append(self._merge_region(key[:, indices], value[:, indices], position[:, indices], scale))
        return tuple(torch.cat([g[i] for g in groups], dim=1) for i in range(5))

    def _spatial_bias(self, mean, variance, target_cell, h, w, scale):
        # Gaussian moment approximation, not an exact pixel-area integral.
        # target_cell is measured in original (un-padded) LR pixel units.
        patch_variance = float((scale*self.ksize)**2)/12
        cov = variance + target_cell[:, None].square()/12 + patch_variance
        cov = cov.clamp_min(1e-8)
        yy = torch.arange(h, device=mean.device, dtype=mean.dtype)[None, None, :, None]
        xx = torch.arange(w, device=mean.device, dtype=mean.dtype)[None, None, None, :]
        dy = yy-mean[..., 0, None, None]
        dx = xx-mean[..., 1, None, None]
        distance = dy.square()/cov[..., 0, None, None]+dx.square()/cov[..., 1, None, None]
        logdet = cov.log().sum(-1)[..., None, None]
        return -.5*(distance+logdet)

    def forward(self, feature, cell=None):
        if feature.ndim != 4:
            raise ValueError('feature must be BCHW')
        if self.geometry:
            if cell is None or cell.ndim != 3 or cell.shape[0] != feature.shape[0] or cell.shape[-1] != 2 or cell.shape[1] == 0:
                raise ValueError('geometry requires cell with shape [B,Q,2]')
            if not torch.isfinite(cell).all() or (cell <= 0).any():
                raise ValueError('cell must be finite and positive')
            if not torch.allclose(cell, cell[:, :1].expand_as(cell)):
                raise ValueError('cell must be constant within each target image')
        if hasattr(torch, 'amp') and hasattr(torch.amp, 'autocast'):
            context = torch.amp.autocast(feature.device.type, enabled=False)
        elif feature.is_cuda:
            context = torch.cuda.amp.autocast(enabled=False)
        else:
            context = nullcontext()
        with context:
            output = self._forward_float(feature.float(), cell)
        return output.to(feature.dtype)

    def _forward_float(self, feature, cell):
        b, c, h, w = feature.shape
        self.last_bank_stats = []
        target_cell = (cell[:, 0].to(feature)*feature.new_tensor([h/2, w/2])
                       if self.geometry else None)
        outputs = []
        for scale in self.scale:
            pad_h, pad_w = (-h) % scale, (-w) % scale
            # Match original reflect padding. Unsupported degenerate sizes
            # are rejected explicitly rather than silently changing padding.
            if pad_h >= h or pad_w >= w:
                raise ValueError('input is too small for original reflect padding at this internal scale')
            padded = F.pad(feature, (0, pad_w, 0, pad_h), 'reflect')
            hp, wp = padded.shape[-2:]
            assembly = self.conv_assembly(padded)
            query = self.conv_match_1(padded)
            kernel = scale*self.ksize
            raw = extract_image_patches(assembly, [kernel, kernel], [scale, scale], [1, 1], 'same')
            ref = self.conv_match_2(F.interpolate(padded, scale_factor=1./scale, mode='bilinear'))
            sh, sw = ref.shape[-2:]
            keys = extract_image_patches(ref, [self.ksize, self.ksize], [1, 1], [1, 1], 'same')
            n = sh*sw
            initial_count = sum(math.ceil(min(self.region_size, sh-y)*
                                          min(self.region_size, sw-x)*self.keep_ratio)
                                for y in range(0, sh, self.region_size)
                                for x in range(0, sw, self.region_size))
            keys = keys.transpose(1, 2).contiguous()
            values = raw.transpose(1, 2).contiguous()
            assert keys.shape[1] == values.shape[1] == n
            # Clamp before sqrt: same original 1e-4 norm floor, but avoids
            # an infinite sqrt derivative at exactly zero feature vectors.
            keys = keys / keys.square().sum(-1, keepdim=True).clamp_min(1e-8).sqrt()
            keys, values, mean, variance, mass = self._aggregate_bank(keys, values, sh, sw, scale)
            query = same_padding(query, [self.ksize, self.ksize], [1, 1], [1, 1])
            batch_outputs = []
            for index in range(b):
                # Strip padding before dynamic convolution, log(mass), and
                # softmax. Aggregates and singletons form ONE candidate bank.
                valid = mass[index] > 0
                key_bank = keys[index, valid]
                value_bank = values[index, valid]
                m = key_bank.shape[0]
                self.last_bank_stats.append(dict(
                    image=index, scale=scale, source=n, initial=initial_count,
                    retained=m, rejected=m-initial_count, retained_ratio=m/n))
                key_kernels = key_bank.reshape(m, ref.shape[1], self.ksize, self.ksize)
                value_kernels = value_bank.reshape(m, c, kernel, kernel)
                logits = F.conv2d(query[index:index+1], key_kernels)*self.softmax_scale
                logits = logits + mass[index, valid].log()[None, :, None, None]
                if self.geometry:
                    geometry_bias = self._spatial_bias(
                        mean[index:index+1, valid], variance[index:index+1, valid],
                        target_cell[index:index+1], hp, wp, scale)
                    logits = logits + F.softplus(self.geometry_log_strength)*geometry_bias
                attention = logits.softmax(dim=1)
                result = F.conv_transpose2d(attention, value_kernels, stride=scale, padding=scale)
                down = self.down if scale == 2 else getattr(self, 'downx%d' % scale)
                result = down(result)/6.
                batch_outputs.append(result[:, :, :h, :w])
            outputs.append(torch.cat(batch_outputs, dim=0))
        return torch.cat(outputs, dim=1)
