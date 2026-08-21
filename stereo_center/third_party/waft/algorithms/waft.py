import torch
import torchvision
import torch.nn as nn
import torch.nn.functional as F
from contextlib import nullcontext

from copy import deepcopy
from timm.layers import Mlp
from einops import rearrange

from model.iterative import fetch_iterative_module
from model.encoder import fetch_feature_encoder
from model.utils import Padder, disp_warp, gaussian_weights


def _timed(timing, name):
    return timing.measure(name) if timing is not None else nullcontext()

def freeze_module(module):
    for p in module.parameters():
        p.requires_grad = False
    for p in module.buffers():
        p.requires_grad = False


def _backward_warp_with_flow(source, target_to_source_flow):
    """Backward-sample source using flow defined on target coordinates."""
    n, _c, h, w = source.shape
    yy, xx = torch.meshgrid(
        torch.arange(h, device=source.device, dtype=source.dtype),
        torch.arange(w, device=source.device, dtype=source.dtype),
        indexing='ij',
    )
    sx = xx.unsqueeze(0) + target_to_source_flow[:, 0]
    sy = yy.unsqueeze(0) + target_to_source_flow[:, 1]
    valid = (
        (sx >= 0.0) & (sx <= w - 1) & (sy >= 0.0) & (sy <= h - 1)
    ).unsqueeze(1)
    grid = torch.stack([
        2.0 * sx / max(w - 1, 1) - 1.0,
        2.0 * sy / max(h - 1, 1) - 1.0,
    ], dim=-1)
    warped = F.grid_sample(
        source, grid, mode='bilinear', padding_mode='zeros', align_corners=True
    )
    return warped, valid


def temporal_warmstart_disparity(
    current_disparity,
    backward_flow,
    temporal_mask,
    group_size,
    carry_disparity=None,
    blend=0.75,
    disparity_abs_tol=3.0,
    disparity_rel_tol=0.15,
):
    """Build a confidence-gated temporal initialization for batched WAFT.

    Samples are partitioned into consecutive temporal groups. Within each
    group, frame ``t`` draws only from frame ``t-1``. The first frame can use
    one carry disparity per group; without a carry its temporal mask is zero.
    """
    if current_disparity.ndim != 4 or current_disparity.shape[1] != 1:
        raise ValueError('current_disparity must have shape (N, 1, H, W)')
    n, _c, h, w = current_disparity.shape
    if group_size <= 0 or n % int(group_size) != 0:
        raise ValueError('group_size must evenly divide the sample batch')
    if backward_flow.ndim != 4 or backward_flow.shape[:2] != (n, 2):
        raise ValueError('backward_flow must have shape (N, 2, H, W)')
    if temporal_mask.ndim != 4 or temporal_mask.shape[:2] != (n, 1):
        raise ValueError('temporal_mask must have shape (N, 1, H, W)')

    flow_h, flow_w = backward_flow.shape[-2:]
    scale_x = w / max(flow_w, 1)
    scale_y = h / max(flow_h, 1)
    flow = F.interpolate(
        backward_flow, size=(h, w), mode='bilinear', align_corners=True
    )
    flow = flow.clone()
    flow[:, 0] *= scale_x
    flow[:, 1] *= scale_y
    mask = F.interpolate(temporal_mask.float(), size=(h, w), mode='bilinear', align_corners=True)

    previous = current_disparity.detach().clone()
    groups = n // int(group_size)
    carry = None
    if carry_disparity is not None:
        carry = carry_disparity
        if carry.ndim == 3:
            carry = carry.unsqueeze(1)
        if carry.ndim != 4 or carry.shape[:2] != (groups, 1):
            raise ValueError('carry_disparity must have shape (groups, H, W)')
        carry_w = carry.shape[-1]
        carry = F.interpolate(carry.float(), size=(h, w), mode='bilinear', align_corners=True)
        carry = carry * (w / max(carry_w, 1))
        carry = carry.to(dtype=current_disparity.dtype, device=current_disparity.device)

    for group in range(groups):
        start = group * int(group_size)
        end = start + int(group_size)
        if end - start > 1:
            previous[start + 1:end] = current_disparity[start:end - 1].detach()
        if carry is None:
            mask[start] = 0.0
        else:
            previous[start] = carry[group]

    prior, in_bounds = _backward_warp_with_flow(previous, flow)
    abs_tol = float(disparity_abs_tol) * scale_x
    threshold = torch.maximum(
        torch.full_like(current_disparity, abs_tol),
        float(disparity_rel_tol) * current_disparity.abs(),
    )
    agreement = (prior - current_disparity).abs() <= threshold
    effective = (
        mask.clamp(0.0, 1.0)
        * in_bounds.to(mask.dtype)
        * agreement.to(mask.dtype)
    )
    weight = float(blend) * effective
    warmstart = current_disparity * (1.0 - weight) + prior * weight
    return warmstart, effective

class WAFT(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.task = cfg.WAFT.ITERATIVE_MODULE.TASK
        self.iters = len(self.task)
        self.n_bins = (int)(cfg.WAFT.LOSS[0].split('_')[-1]) + 1
        self.encoder, self.enc_dim, self.factor = fetch_feature_encoder(cfg.WAFT.FEATURE_ENCODER)
        self.hidden_dim = self.enc_dim
        self.prop_decoder = fetch_iterative_module(cfg.WAFT.ITERATIVE_MODULE.PROP_ITER, input_dim=self.hidden_dim)
        self.prop_proj = Mlp(self.enc_dim*2, self.hidden_dim, self.hidden_dim, use_conv=True)
        self.delta_decoder = fetch_iterative_module(cfg.WAFT.ITERATIVE_MODULE.DELTA_ITER, input_dim=self.hidden_dim)
        self.delta_proj = Mlp(self.enc_dim*2+self.hidden_dim+1, self.hidden_dim, self.hidden_dim, use_conv=True)
        
        self.max_disp = cfg.WAFT.MAX_DISP
        self.delta_mask_head = Mlp(self.hidden_dim, self.hidden_dim, 4*9, use_conv=True)
        self.delta_dist_head = Mlp(self.hidden_dim, self.hidden_dim, 4, use_conv=True)
        self.delta_disp_head = Mlp(self.hidden_dim, self.hidden_dim, 1, use_conv=True)
        self.prop_mask_head = Mlp(self.hidden_dim, self.hidden_dim, 4*9, use_conv=True)
        self.prop_bins_head = Mlp(self.hidden_dim, self.hidden_dim, self.n_bins, use_conv=True)

    def normalize_image(self, img):
        '''
        @img: (B,C,H,W) in range 0-255, RGB order
        '''
        tf = torchvision.transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225], inplace=False)
        return tf(img/255.0).contiguous()

    def convex_upsample(self, info, mask):
        N, C, H, W = info.shape
        mask = mask.view(N, 1, 9, 2, 2, H, W)
        mask = torch.softmax(mask, dim=2)
        up_info = F.unfold(info, [3, 3], padding=1)
        up_info = up_info.view(N, C, 9, 1, 1, H, W)
        up_info = torch.sum(mask * up_info, dim=2)
        up_info = up_info.permute(0, 1, 4, 2, 5, 3)
        return up_info.reshape(N, C, 2*H, 2*W)
    
    def forward(self, sample, disp_init=None, temporal_init=None, timing=None):
        """ Estimate disparity between pair of frames """
        if disp_init is not None and temporal_init is not None:
            raise ValueError('disp_init and temporal_init are mutually exclusive')
        output = {}
        with _timed(timing, "normalize"):
            image1 = self.normalize_image(sample['img1'])
            image2 = self.normalize_image(sample['img2'])
        with _timed(timing, "pad_input"):
            padder = Padder(image1.shape, factor=self.factor)
            image1 = padder.pad(image1)
            image2 = padder.pad(image2)

        with _timed(timing, "feature_encoder"):
            fmap1, fmap2, net = self.encoder(torch.stack([image1, image2], dim=1))
        n, _, h, w = fmap1.shape

        with _timed(timing, "initial_disparity"):
            idx_bins_2x = torch.linspace(0, self.max_disp/2, self.n_bins, device=fmap1.device, dtype=fmap1.dtype).view(1, self.n_bins, 1, 1)
            idx_bins_1x = torch.linspace(0, self.max_disp/1, self.n_bins, device=fmap1.device, dtype=fmap1.dtype).view(1, self.n_bins, 1, 1)

            prop_hidden = self.prop_proj(torch.cat([fmap1, fmap2], dim=1))
            prop_hidden = self.prop_decoder(prop_hidden)
            prob_mask = .25 * self.prop_mask_head(prop_hidden)
            prob_bins = self.prop_bins_head(prop_hidden)
            prob_up = self.convex_upsample(prob_bins, prob_mask)
            output['init'] = padder.unpad(prob_up)
            prob_bins = F.softmax(prob_bins, dim=1)
            disp = torch.sum(prob_bins * idx_bins_2x, dim=1, keepdim=True)

            if disp_init is not None:
                disp = padder.pad(disp_init.unsqueeze(1))
                disp = F.interpolate(disp, scale_factor=0.5, mode='bilinear', align_corners=True) * 0.5

        if temporal_init is not None:
            with _timed(timing, "temporal_initialization"):
                temporal_flow = padder.pad(
                    temporal_init['backward_flow'].to(device=disp.device, dtype=disp.dtype)
                )
                temporal_mask = padder.pad(
                    temporal_init['mask'].to(device=disp.device, dtype=disp.dtype)
                )
                carry = temporal_init.get('carry_disparity')
                if carry is not None:
                    carry = padder.pad(carry.to(device=disp.device, dtype=disp.dtype))
                disp, temporal_valid = temporal_warmstart_disparity(
                    disp,
                    temporal_flow,
                    temporal_mask,
                    group_size=int(temporal_init['group_size']),
                    carry_disparity=carry,
                    blend=float(temporal_init.get('blend', 0.75)),
                    disparity_abs_tol=float(temporal_init.get('disparity_abs_tol', 3.0)),
                    disparity_rel_tol=float(temporal_init.get('disparity_rel_tol', 0.15)),
                )
                output['temporal_valid_mask'] = temporal_valid

        delta_disp_preds = []
        delta_info_preds = []
        for itr in range(self.iters):
            with _timed(timing, f"iteration_{itr + 1}"):
                disp = disp.detach()
                warped_fmap2 = disp_warp(fmap2, disp, padding_mode='zeros')
                net = self.delta_proj(torch.cat([fmap1, warped_fmap2, net, disp], dim=1))
                net = self.delta_decoder(net)
                info = self.delta_dist_head(net)
                delta_disp = self.delta_disp_head(net)
                mask = .25 * self.delta_mask_head(net)
                disp = disp + delta_disp
                disp_up = self.convex_upsample(disp * 2, mask)
                info_up = self.convex_upsample(info, mask)
                delta_disp_preds.append(disp_up)
                delta_info_preds.append(info_up)

        with _timed(timing, "unpad_outputs"):
            output['delta_disp_preds'] = [padder.unpad(disp) for disp in delta_disp_preds]
            output['delta_info_preds'] = [padder.unpad(info) for info in delta_info_preds]
        with _timed(timing, "finalize_disparity"):
            if self.iters > 0:
                disp_final = output['delta_disp_preds'][-1].squeeze(1)
                output['disp_pred'] = disp_final
            else:
                disp_final = torch.sum(F.softmax(output['init'], dim=1) * idx_bins_1x, dim=1)
                output['disp_pred'] = disp_final
        
        return output

    def inference(self, sample, size=None, factor=1.0, disp_init=None, timing=None):
        sample = {
            'img1': F.interpolate(sample['img1'], scale_factor=factor, mode='bilinear', align_corners=True),
            'img2': F.interpolate(sample['img2'], scale_factor=factor, mode='bilinear', align_corners=True)
        }
        disp_init = None if disp_init is None else F.interpolate(disp_init.unsqueeze(1), scale_factor=factor, mode='bilinear', align_corners=True).squeeze(1) * factor

        if size is None:
            with _timed(timing, "inference_forward"):
                output = self.forward(sample, disp_init=disp_init, timing=timing)
            for k in output.keys():
                if 'disp' in k:
                    ratio = 1/factor
                else:
                    ratio = 1
                if isinstance(output[k], list):
                    for i in range(len(output[k])):
                        output[k][i] = F.interpolate(output[k][i], scale_factor=1/factor, mode='bilinear', align_corners=True) * ratio
                else:
                    if output[k].dim() == 4:
                        output[k] = F.interpolate(output[k], scale_factor=1/factor, mode='bilinear', align_corners=True)
                    else:
                        output[k] = F.interpolate(output[k][:, None], scale_factor=1/factor, mode='bilinear', align_corners=True).squeeze(1) * ratio
            return output
        
        img1 = sample['img1']
        img2 = sample['img2']
        padder = Padder(img1.shape, size=size)
        img1 = padder.pad(img1)
        img2 = padder.pad(img2)
        disp_init = None if disp_init is None else padder.pad(disp_init)
        hstep = size[0] - 16
        wstep = size[1] - 16
        gau = gaussian_weights(size[0], size[1], device=img1.device).view(1, size[0], size[1])
        b, _, h, w = img1.shape
        weights = torch.zeros((b, h, w), device=img1.device, dtype=img1.dtype)
        output = {}
        for idx_h in range(0, h, hstep):
            for idx_w in range(0, w, wstep):
                rh = min(idx_h + size[0], h)
                rw = min(idx_w + size[1], w)
                lh = rh - size[0]
                lw = rw - size[1]
                sample_patch = {}
                sample_patch['img1'] = img1[:, :, lh:rh, lw:rw]
                sample_patch['img2'] = img2[:, :, lh:rh, lw:rw]
                disp_init_patch = None if disp_init is None else disp_init[:, lh:rh, lw:rw]
                output_patch = self.forward(sample_patch, disp_init=disp_init_patch, timing=timing)
                for k in output_patch.keys():
                    if k not in output:
                        if isinstance(output_patch[k], list):
                            output[k] = []
                            for t in output_patch[k]:
                                if t.dim() == 4:
                                    output[k].append(torch.zeros((b, t.shape[1], h, w), device=t.device, dtype=t.dtype))
                                else:
                                    output[k].append(torch.zeros((b, h, w), device=t.device, dtype=t.dtype))
                        else:
                            t = output_patch[k]
                            if t.dim() == 4:
                                output[k] = torch.zeros((b, t.shape[1], h, w), device=t.device, dtype=t.dtype)
                            else:
                                output[k] = torch.zeros((b, h, w), device=t.device, dtype=t.dtype)
                    if isinstance(output_patch[k], list):
                        for i in range(len(output_patch[k])):
                            output[k][i][..., lh:rh, lw:rw] += output_patch[k][i] * gau
                    else:
                        output[k][..., lh:rh, lw:rw] += output_patch[k] * gau
                weights[:, lh:rh, lw:rw] += gau
        
        for k in output.keys():
            if 'disp' in k:
                ratio = 1/factor
            else:
                ratio = 1
            if isinstance(output[k], list):
                for i in range(len(output[k])):
                    output[k][i] = padder.unpad(output[k][i] / weights)
                    output[k][i] = F.interpolate(output[k][i], scale_factor=1/factor, mode='bilinear', align_corners=True) * ratio
            else:
                output[k] = padder.unpad(output[k] / weights)
                if output[k].dim() == 4:
                    output[k] = F.interpolate(output[k], scale_factor=1/factor, mode='bilinear', align_corners=True)
                else:
                    output[k] = F.interpolate(output[k][:, None], scale_factor=1/factor, mode='bilinear', align_corners=True).squeeze(1) * ratio
        
        return output
    
    def heirarchical_inference(self, sample, size=None, factor_list=None, timing=None):
        output = {}
        disp_init = None
        for i in range(len(factor_list)):
            with _timed(timing, f"hierarchical_scale_{factor_list[i]}"):
                output = self.inference(sample, size=size, factor=factor_list[i], disp_init=disp_init, timing=timing)
            disp_init = output['disp_pred']

        return output
