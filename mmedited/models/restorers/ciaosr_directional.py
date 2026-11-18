"""Baseline optimization plus scheduled, frozen-teacher relation supervision."""
import torch

from mmedited.models.restorers.ciaosr import CiaoSR
from mmedited.models.common.directional_detail import relation_distillation_loss
from mmedited.models.common.dinov3_teacher import DinoV3Teacher


class CiaoSRDirectional(CiaoSR):
    def __init__(self, *args, teacher=None, **kwargs):
        super().__init__(*args, **kwargs)
        settings = dict(teacher or {})
        self.teacher_enabled = settings.pop('enabled', False)
        self.teacher_loss_weight = float(settings.pop('loss_weight', .01))
        self.teacher_interval = int(settings.pop('interval', 4))
        self.teacher_warmup = int(settings.pop('warmup', 500))
        self.teacher_ramp = int(settings.pop('ramp', 1000))
        self.teacher_max_grid = int(settings.pop('max_grid', 8))
        if self.teacher_loss_weight <= 0 or self.teacher_interval < 1:
            raise ValueError('teacher loss_weight must be positive and interval >=1')
        if min(self.teacher_warmup, self.teacher_ramp) < 0 or self.teacher_max_grid < 2:
            raise ValueError('invalid teacher schedule/grid')
        if self.teacher_enabled and self.generator.directional_detail is None:
            raise ValueError('teacher requires an enabled directional branch')
        self.teacher_settings = settings
        self.teacher_client = None  # Plain Python object, not registered module.
        self.register_buffer('guidance_step', torch.zeros((), dtype=torch.long))

    def train_step(self, data_batch, optimizer):
        coord, cell = data_batch['coord'], data_batch['cell']
        rgb, gt = data_batch['lq'], data_batch['gt']
        self.lq_mean, self.lq_std = self.lq_mean.to(rgb), self.lq_std.to(rgb)
        self.gt_mean, self.gt_std = self.gt_mean.to(gt), self.gt_std.to(gt)
        lq = (rgb - self.lq_mean) / self.lq_std
        target = (gt - self.gt_mean) / self.gt_std
        step = int(self.guidance_step.item())
        guided = (self.teacher_enabled and step >= self.teacher_warmup
                  and (step - self.teacher_warmup) % self.teacher_interval == 0)
        if guided:
            pred, structure = self.generator(lq, coord, cell, return_aux=True)
        else:
            pred = self.generator(lq, coord, cell)
        losses = {'loss_pix': self.pixel_loss(pred, target)}
        if guided:
            if self.teacher_client is None:
                self.teacher_client = DinoV3Teacher(**self.teacher_settings)
            teacher_map = self.teacher_client(rgb)
            ramp = min(1., (step - self.teacher_warmup + 1) / max(1, self.teacher_ramp))
            losses['loss_dino_relation'] = self.teacher_loss_weight * ramp * relation_distillation_loss(
                structure, teacher_map, max_grid=self.teacher_max_grid)
        loss, logs = self.parse_losses(losses)
        if not torch.isfinite(loss):
            raise FloatingPointError('Non-finite training loss; optimizer step aborted')
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        self.guidance_step.add_(1)
        logs.pop('loss', None)
        # Consistent log keys on guided and unguided steps; same schedule on DDP ranks.
        logs.setdefault('loss_dino_relation', 0.)
        logs['dino_active'] = float(guided)
        return dict(log_vars=logs, num_samples=len(gt),
                    results=dict(lq=lq.detach().cpu(), gt=target.detach().cpu(),
                                 output=pred.detach().cpu()))
