"""Experimental decoder; the ordinary-mean baseline file is unchanged."""
import torch
from torch.nn import functional as F
from mmedit.datasets.pipelines.utils import make_coord
from mmedited.models.backbones.sr_backbones.ciaosr_patch_kv_net import LocalImplicitSRRDN as BaselineRDN
from mmedited.models.common.directional_detail import DirectionalDetail


class DirectionalImplicitSRRDN(BaselineRDN):
    def __init__(self, *args, directional=None, **kwargs):
        settings = dict(directional or {})
        enabled = settings.pop('enabled', False)
        # Read before the baseline constructor mutates component dimensions.
        encoder = kwargs.get('encoder', args[0] if args else None)
        if encoder is None:
            raise ValueError('encoder configuration is required')
        channels = encoder['mid_channels']
        super().__init__(*args, **kwargs)
        if enabled and self.local_size != 2:
            raise ValueError('directional reconstruction requires local_size=2')
        self.directional_detail = DirectionalDetail(channels, **settings) if enabled else None

    def _prepare_attention_features(self, features, cell):
        prepared = super()._prepare_attention_features(features, cell)
        if self.directional_detail is None:
            return prepared
        return [entry + (self.directional_detail.structure(feature),)
                for entry, feature in zip(prepared, features)]

    def forward(self, x, coord, cell, test_mode=False, return_aux=False):
        if not return_aux:
            return super().forward(x, coord, cell, test_mode=test_mode)
        if self.directional_detail is None or test_mode:
            raise ValueError('return_aux is only supported for enabled training')
        features = self.gen_feature(x)
        prepared = self._prepare_attention_features(features, cell)
        pred = self.query_rgb(features, coord, cell, prepared)
        pred = pred + F.grid_sample(
            x, coord.flip(-1).unsqueeze(1), mode='bilinear',
            padding_mode='border', align_corners=False)[:, :, 0, :].permute(0, 2, 1)
        # Return tensors through forward for DDP graph discovery; no mutable cache.
        return pred, prepared[0][3]

    def query_rgb(self, features, coord, scale=None, prepared=None):
        """Query RGB value of GT.

        Copyright (c) 2020, Yinbo Chen, under BSD 3-Clause License.

        Args:
            feature (Tensor): encoded feature.
            coord (Tensor): coord tensor, shape (BHW, 2).

        Returns:
            result (Tensor): (part of) output.
        """

        res_features = []
        if prepared is None:
            prepared = self._prepare_attention_features(features, scale)
        for feature_index, feature in enumerate(features):
            
            B, C, H, W = feature.shape      #[16, 64, 48, 48]

            entry = prepared[feature_index]
            feat_q, feat_k, feat_v = entry[:3]
            structure_map = entry[3] if self.directional_detail is not None else None


            # query
            query = F.grid_sample(feat_q, coord.flip(-1).unsqueeze(1), mode='nearest', 
                        align_corners=False).permute(0, 3, 2, 1).contiguous()       #[16, 2304, 1, 576]

            feat_coord = make_coord(feature.shape[-2:], flatten=False).permute(2, 0, 1) \
                            .unsqueeze(0).expand(B, 2, *feature.shape[-2:])         #[16, 2, 48, 48]
            feat_coord = feat_coord.to(coord)

            if self.local_size == 1:
                v_lst = [(0, 0)]
            else:
                v_lst = [(i,j) for i in range(-1, 2, 4-self.local_size) for j in range(-1, 2, 4-self.local_size)]
            eps_shift = 1e-6
            preds_k, preds_v = [], []
            relative_offsets = []
            
            for v in v_lst:
                vx, vy = v[0], v[1]
                # project to LR field
                tx = ((H - 1) / (1 - scale[:,0,0])).view(B,  1)     # [16, 1]
                ty = ((W - 1) / (1 - scale[:,0,1])).view(B,  1)     # [16, 1]
                rx = (2*abs(vx) -1) / tx if vx != 0 else 0          # [16, 1]
                ry = (2*abs(vy) -1) / ty if vy != 0 else 0          # [16, 1]
                
                bs, q = coord.shape[:2]     
                coord_ = coord.clone()  # [16, 2304, 2]
                if vx != 0:
                    coord_[:, :, 0] += vx /abs(vx) * rx + eps_shift  # [16, 2304]
                if vy != 0:
                    coord_[:, :, 1] += vy /abs(vy) * ry + eps_shift  # [16, 2304]
                coord_.clamp_(-1 + 1e-6, 1 - 1e-6)

                # key and value
                key = F.grid_sample(feat_k, coord_.flip(-1).unsqueeze(1), mode='nearest', 
                    align_corners=False)[:, :, 0, :].permute(0, 2, 1).contiguous()          #[16, 2304, 576]
                value = F.grid_sample(feat_v, coord_.flip(-1).unsqueeze(1), mode='nearest', 
                    align_corners=False)[:, :, 0, :].permute(0, 2, 1).contiguous()          #[16, 2304, 576]

                #Interpolate K to HR resolution
                coord_k = F.grid_sample(feat_coord, coord_.flip(-1).unsqueeze(1),
                    mode='nearest', align_corners=False)[:, :, 0, :].permute(0, 2, 1)       #[16, 2304, 2]

                Q, K = coord, coord_k   #[16, 2304, 2]
                rel = Q - K             #[16, 2304, 2]
                rel[:, :, 0] *= feature.shape[-2]   # without mul
                rel[:, :, 1] *= feature.shape[-1]
                relative_offsets.append(rel)
                inp = rel   #[16, 2304, 2]

                scale_ = scale.clone()      #[16, 2304, 2]
                scale_[:, :, 0] *= feature.shape[-2]
                scale_[:, :, 1] *= feature.shape[-1]

                inp_v = torch.cat([value, inp, scale_], dim=-1)   #[16, 2304, 580]
                inp_k = torch.cat([key, inp, scale_], dim=-1)     #[16, 2304, 580]
                    

                inp_k = inp_k.contiguous().view(bs * q, -1)
                inp_v = inp_v.contiguous().view(bs * q, -1)

                weight_k = self.imnet_k(inp_k).view(bs, q, -1).contiguous()   #[16, 2304, 576]
                pred_k = (key * weight_k).view(bs, q, -1)                     #[16, 2304, 576]
                
                weight_v = self.imnet_v(inp_v).view(bs, q, -1).contiguous()   #[16, 2304, 576]
                pred_v = (value * weight_v).view(bs, q, -1)                   #[16, 2304, 576]

                preds_v.append(pred_v)
                preds_k.append(pred_k)

            preds_k = torch.stack(preds_k, dim=-1)                      # [16, 2304, 576, 4]
            preds_v = torch.stack(preds_v, dim=-2)                      # [16, 2304, 4, 576]

            attn = (query @ preds_k)                                    # [16, 2304, 1, 4]
            x = ((attn/self.softmax_scale).softmax(dim=-1) @ preds_v)   # [16, 2304, 1, 576]
            if self.directional_detail is not None:
                structure = F.grid_sample(
                    structure_map, coord.flip(-1).unsqueeze(1),
                    mode='nearest', align_corners=False)[:, :, 0, :].permute(0, 2, 1)
                cell_lr = scale * scale.new_tensor([H, W])
                detail = self.directional_detail(
                    preds_v, structure, torch.stack(relative_offsets, dim=-2), cell_lr)
                x = x + detail.unsqueeze(-2)
            x = x.view(bs*q, -1)                                        # [16*2304, 576]

            res_features.append(x) 

        result = torch.cat(res_features, dim=-1)    # [16, 2304, 576x2]
        result = self.imnet_q(result)               # [16, 2304, 3]
        result = result.view(bs, q, -1)

        return result
