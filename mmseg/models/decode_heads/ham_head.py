import torch
import torch.nn as nn
import torch.nn.functional as F
from mmcv.cnn import ConvModule

from mmseg.ops import resize
from ..builder import HEADS, build_loss
from .decode_head import BaseDecodeHead

try:
    from pytorch_wavelets import DWTForward, DWTInverse
except ImportError:
    DWTForward = None
    DWTInverse = None


class _MatrixDecomposition2DBase(nn.Module):
    def __init__(self, args=dict()):
        super().__init__()

        self.spatial = args.setdefault('SPATIAL', True)

        self.S = args.setdefault('MD_S', 1)
        self.D = args.setdefault('MD_D', 512)
        self.R = args.setdefault('MD_R', 64)

        self.train_steps = args.setdefault('TRAIN_STEPS', 6)
        self.eval_steps = args.setdefault('EVAL_STEPS', 7)

        self.inv_t = args.setdefault('INV_T', 100)
        self.eta = args.setdefault('ETA', 0.9)

        self.rand_init = args.setdefault('RAND_INIT', True)

        print('spatial', self.spatial)
        print('S', self.S)
        print('D', self.D)
        print('R', self.R)
        print('train_steps', self.train_steps)
        print('eval_steps', self.eval_steps)
        print('inv_t', self.inv_t)
        print('eta', self.eta)
        print('rand_init', self.rand_init)

    def _build_bases(self, B, S, D, R, cuda=False):
        raise NotImplementedError

    def local_step(self, x, bases, coef):
        raise NotImplementedError

    # @torch.no_grad()
    def local_inference(self, x, bases):
        # (B * S, D, N)^T @ (B * S, D, R) -> (B * S, N, R)
        coef = torch.bmm(x.transpose(1, 2), bases)
        coef = F.softmax(self.inv_t * coef, dim=-1)

        steps = self.train_steps if self.training else self.eval_steps
        for _ in range(steps):
            bases, coef = self.local_step(x, bases, coef)

        return bases, coef

    def compute_coef(self, x, bases, coef):
        raise NotImplementedError

    def forward(self, x, return_bases=False):
        B, C, H, W = x.shape

        # (B, C, H, W) -> (B * S, D, N)
        if self.spatial:
            D = C // self.S
            N = H * W
            x = x.view(B * self.S, D, N)
        else:
            D = H * W
            N = C // self.S
            x = x.view(B * self.S, N, D).transpose(1, 2)

        if not self.rand_init and not hasattr(self, 'bases'):
            bases = self._build_bases(1, self.S, D, self.R, cuda=True)
            self.register_buffer('bases', bases)

        # (S, D, R) -> (B * S, D, R)
        if self.rand_init:
            bases = self._build_bases(B, self.S, D, self.R, cuda=True)
        else:
            bases = self.bases.repeat(B, 1, 1)

        bases, coef = self.local_inference(x, bases)

        # (B * S, N, R)
        coef = self.compute_coef(x, bases, coef)

        # (B * S, D, R) @ (B * S, N, R)^T -> (B * S, D, N)
        x = torch.bmm(bases, coef.transpose(1, 2))

        # (B * S, D, N) -> (B, C, H, W)
        if self.spatial:
            x = x.view(B, C, H, W)
        else:
            x = x.transpose(1, 2).view(B, C, H, W)

        # (B * H, D, R) -> (B, H, N, D)
        bases = bases.view(B, self.S, D, self.R)

        return x


class NMF2D(_MatrixDecomposition2DBase):
    def __init__(self, args=dict()):
        super().__init__(args)

        self.inv_t = 1

    def _build_bases(self, B, S, D, R, cuda=False):
        if cuda:
            bases = torch.rand((B * S, D, R)).cuda()
        else:
            bases = torch.rand((B * S, D, R))

        bases = F.normalize(bases, dim=1)

        return bases

    # @torch.no_grad()
    def local_step(self, x, bases, coef):
        # (B * S, D, N)^T @ (B * S, D, R) -> (B * S, N, R)
        numerator = torch.bmm(x.transpose(1, 2), bases)
        # (B * S, N, R) @ [(B * S, D, R)^T @ (B * S, D, R)] -> (B * S, N, R)
        denominator = coef.bmm(bases.transpose(1, 2).bmm(bases))
        # Multiplicative Update
        coef = coef * numerator / (denominator + 1e-6)

        # (B * S, D, N) @ (B * S, N, R) -> (B * S, D, R)
        numerator = torch.bmm(x, coef)
        # (B * S, D, R) @ [(B * S, N, R)^T @ (B * S, N, R)] -> (B * S, D, R)
        denominator = bases.bmm(coef.transpose(1, 2).bmm(coef))
        # Multiplicative Update
        bases = bases * numerator / (denominator + 1e-6)

        return bases, coef

    def compute_coef(self, x, bases, coef):
        # (B * S, D, N)^T @ (B * S, D, R) -> (B * S, N, R)
        numerator = torch.bmm(x.transpose(1, 2), bases)
        # (B * S, N, R) @ (B * S, D, R)^T @ (B * S, D, R) -> (B * S, N, R)
        denominator = coef.bmm(bases.transpose(1, 2).bmm(bases))
        # multiplication update
        coef = coef * numerator / (denominator + 1e-6)

        return coef


class Hamburger(nn.Module):
    def __init__(self,
                 ham_channels=512,
                 ham_kwargs=dict(),
                 norm_cfg=None,
                 **kwargs):
        super().__init__()

        self.ham_in = ConvModule(
            ham_channels,
            ham_channels,
            1,
            norm_cfg=None,
            act_cfg=None
        )

        self.ham = NMF2D(ham_kwargs)

        self.ham_out = ConvModule(
            ham_channels,
            ham_channels,
            1,
            norm_cfg=norm_cfg,
            act_cfg=None)

    def forward(self, x):
        enjoy = self.ham_in(x)
        enjoy = F.relu(enjoy, inplace=True)
        enjoy = self.ham(enjoy)
        enjoy = self.ham_out(enjoy)
        ham = F.relu(x + enjoy, inplace=True)

        return ham


class MultiScaleHybridCrossBlock(nn.Module):
    def __init__(self, in_channels, out_channels, bias=True, activation=nn.ReLU(inplace=True)):
        super().__init__()
        k3 = 3
        k5 = 5
        self.conv3_stage1 = nn.Conv2d(in_channels, out_channels, k3, padding=k3 // 2, bias=bias)
        self.conv5_stage1 = nn.Conv2d(in_channels, out_channels, k5, padding=k5 // 2, bias=bias)
        self.conv3_stage2 = nn.Conv2d(out_channels, out_channels, k3, padding=k3 // 2, bias=bias)
        self.conv5_stage2 = nn.Conv2d(out_channels, out_channels, k5, padding=k5 // 2, bias=bias)
        self.fusion3 = nn.Conv2d(in_channels * 3, out_channels, 1, bias=True)
        self.fusion5 = nn.Conv2d(in_channels * 3, out_channels, 1, bias=True)
        self.bottleneck_fusion = nn.Conv2d(in_channels * 3 + out_channels * 2, out_channels, 1, bias=True)
        self.activation = activation

    def forward(self, x):
        identity = x

        feat3_stage1 = self.activation(self.conv3_stage1(identity))
        feat3_stage1 = feat3_stage1 + identity

        feat5_stage1 = self.activation(self.conv5_stage1(identity))
        feat5_stage1 = feat5_stage1 + identity

        stage1_concat = torch.cat([identity, feat3_stage1, feat5_stage1], dim=1)
        fusion_feat3 = self.fusion3(stage1_concat)
        fusion_feat5 = self.fusion5(stage1_concat)

        feat3_stage2 = self.activation(self.conv3_stage2(fusion_feat3))
        feat5_stage2 = self.activation(self.conv5_stage2(fusion_feat5))

        stage2_concat = torch.cat(
            [identity, feat3_stage1, feat5_stage1, feat3_stage2, feat5_stage2],
            dim=1)
        out = self.bottleneck_fusion(stage2_concat)
        out = out + identity
        return out


class PFESA(nn.Module):
    """Parameter-free edge-structure attention."""

    def __init__(self, base_ratio=0.1, eps=1e-5):
        super().__init__()
        self.activation = nn.Sigmoid()
        self.base_ratio = base_ratio
        self.eps = eps

    def _edge_attention(self, high_freq_feature):
        spatial_mean = high_freq_feature.mean(dim=[2, 3], keepdim=True)
        squared_deviation = (high_freq_feature - spatial_mean).pow(2)
        feature_variance = high_freq_feature.var(dim=[2, 3], keepdim=True)
        return squared_deviation / (feature_variance + self.eps)

    def _structure_attention(self, low_freq_feature):
        low_freq_energy = low_freq_feature.pow(2)
        low_freq_energy_mean = low_freq_energy.mean(dim=[2, 3], keepdim=True)
        low_freq_energy_var = low_freq_energy.var(dim=[2, 3], keepdim=True)
        structure_attention = (
            low_freq_energy - low_freq_energy_mean
        ) / (low_freq_energy_var + self.eps)
        return self.activation(structure_attention)

    def _create_low_freq_mask(self, height, width, device, dtype):
        mask_ratio = self.base_ratio * min(height, width) / max(height, width)
        y_coords = torch.linspace(-1, 1, height, device=device, dtype=dtype)
        x_coords = torch.linspace(-1, 1, width, device=device, dtype=dtype)
        grid_y, grid_x = torch.meshgrid(y_coords, x_coords, indexing='ij')
        return torch.exp(-(grid_y.pow(2) + grid_x.pow(2)) /
                         (2 * mask_ratio ** 2))

    def forward(self, x):
        _, _, height, width = x.shape
        freq_feature = torch.fft.fftn(x, dim=(-2, -1))
        freq_feature = torch.fft.fftshift(freq_feature, dim=(-2, -1))

        low_freq_mask = self._create_low_freq_mask(
            height, width, freq_feature.device, x.dtype)
        low_freq_spectrum = freq_feature * low_freq_mask
        low_freq_feature = torch.abs(
            torch.fft.ifftn(low_freq_spectrum, dim=(-2, -1)))
        structure_attention = self._structure_attention(low_freq_feature)

        high_freq_spectrum = freq_feature * (1 - low_freq_mask)
        high_freq_feature = torch.abs(
            torch.fft.ifftn(high_freq_spectrum, dim=(-2, -1)))
        edge_attention = self._edge_attention(high_freq_feature)

        fused_attention = self.activation(structure_attention + edge_attention)
        return fused_attention * x


class NightSceneStrongDetailGate(nn.Module):
    """Night-scene oriented gate for stage1 detail filtering.

    The gate jointly models:
    1. stage1 fine details,
    2. stage2 semantic guidance,
    3. their disagreement map |stage1-stage2|.

    It uses local depthwise context plus spatial/channel gating, while
    preserving a soft semantic prior from stage2 to avoid over-suppressing
    weak nighttime boundaries.
    """

    def __init__(self, channels, reduction=4):
        super().__init__()
        hidden_channels = max(channels // reduction, 8)
        self.fuse = nn.Sequential(
            nn.Conv2d(channels * 3, channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(channels),
            nn.ReLU(inplace=True))
        self.local_refine = nn.Sequential(
            nn.Conv2d(
                channels,
                channels,
                kernel_size=3,
                padding=1,
                groups=channels,
                bias=False),
            nn.BatchNorm2d(channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels, channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(channels),
            nn.ReLU(inplace=True))
        self.semantic_gate = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=1, bias=True),
            nn.Sigmoid())
        self.channel_gate = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(channels, hidden_channels, kernel_size=1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden_channels, channels, kernel_size=1, bias=True),
            nn.Sigmoid())
        self.spatial_gate = nn.Sequential(
            nn.Conv2d(4, hidden_channels, kernel_size=3, padding=1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden_channels, 1, kernel_size=1, bias=True),
            nn.Sigmoid())

    def forward(self, stage1_feat, stage2_feat):
        detail_disagreement = torch.abs(stage1_feat - stage2_feat)
        fused_feat = self.fuse(
            torch.cat([stage1_feat, stage2_feat, detail_disagreement], dim=1))
        fused_feat = self.local_refine(fused_feat)

        semantic_gate = self.semantic_gate(stage2_feat)
        channel_gate = self.channel_gate(fused_feat)

        spatial_descriptor = torch.cat([
            torch.mean(fused_feat, dim=1, keepdim=True),
            torch.max(fused_feat, dim=1, keepdim=True)[0],
            torch.mean(detail_disagreement, dim=1, keepdim=True),
            torch.max(detail_disagreement, dim=1, keepdim=True)[0],
        ], dim=1)
        spatial_gate = self.spatial_gate(spatial_descriptor)

        # Keep the gate soft to avoid suppressing weak nighttime boundaries.
        gate = semantic_gate
        gate = gate * (0.5 + 0.5 * channel_gate)
        gate = gate * (0.5 + 0.5 * spatial_gate)
        return gate


class DiscreteWaveletTransform2D(nn.Module):
    def __init__(self, wavelet_type='haar', padding_mode='zero'):
        super().__init__()
        if DWTForward is None:
            raise ImportError(
                'pytorch_wavelets is required for '
                'LowFrequencyGuidedFeaturePurification.')
        self.forward_wavelet_transform = DWTForward(
            J=1, wave=wavelet_type, mode=padding_mode)

    def forward(self, x):
        batch_size, _, height, width = x.shape
        with torch.cuda.amp.autocast(enabled=False):
            if x.dtype != torch.float32:
                x = x.float()
            low_freq_feature, high_freq_list = self.forward_wavelet_transform(x)

        high_freq_feature = high_freq_list[0].transpose(1, 2).reshape(
            batch_size,
            -1,
            high_freq_list[0].shape[3],
            high_freq_list[0].shape[4])
        wavelet_feature = torch.cat((low_freq_feature, high_freq_feature), dim=1)
        return F.interpolate(
            wavelet_feature,
            size=(height // 2, width // 2),
            mode='bilinear',
            align_corners=False)


class InverseDiscreteWaveletTransform2D(nn.Module):
    def __init__(self, wavelet_type='haar', padding_mode='zero'):
        super().__init__()
        if DWTInverse is None:
            raise ImportError(
                'pytorch_wavelets is required for '
                'LowFrequencyGuidedFeaturePurification.')
        self.inverse_wavelet_transform = DWTInverse(
            wave=wavelet_type, mode=padding_mode)

    def forward(self, low_frequency_feature, high_frequency_feature):
        batch_size, channels, half_height, half_width = \
            low_frequency_feature.shape
        high_frequency_feature = high_frequency_feature.reshape(
            batch_size, channels, 3, half_height, half_width)

        with torch.cuda.amp.autocast(enabled=False):
            reconstructed_feature = self.inverse_wavelet_transform(
                (low_frequency_feature, [high_frequency_feature.float()]))

        return F.interpolate(
            reconstructed_feature,
            size=(2 * half_height, 2 * half_width),
            mode='bilinear',
            align_corners=False)


class SpatialAttentionMapGenerator(nn.Module):
    def __init__(self, kernel_size=7, use_bn_before_sigmoid=False):
        super().__init__()
        assert kernel_size in (3, 7)
        padding = 3 if kernel_size == 7 else 1
        self.use_bn_before_sigmoid = use_bn_before_sigmoid
        self.spatial_context_conv = nn.Conv2d(
            2, 1, kernel_size=kernel_size, padding=padding, bias=False)
        if use_bn_before_sigmoid:
            self.attention_bn = nn.BatchNorm2d(1)
            self.attention_bn.bias.data.fill_(0)
            self.attention_bn.bias.requires_grad = False
        self.activation = nn.Sigmoid()

    def forward(self, x):
        average_response_map = torch.mean(x, dim=1, keepdim=True)
        maximum_response_map, _ = torch.max(x, dim=1, keepdim=True)
        spatial_descriptor = torch.cat(
            [average_response_map, maximum_response_map], dim=1)
        attention_logit = self.spatial_context_conv(spatial_descriptor)
        if self.use_bn_before_sigmoid:
            attention_logit = self.attention_bn(attention_logit)
        return self.activation(attention_logit)


class LearnableGaussianSmoothingBank(nn.Module):
    def __init__(self, kernel_size, num_filters, num_channels):
        super().__init__()
        self.kernel_size = kernel_size
        self.num_filters = num_filters
        self.num_channels = num_channels
        self.padding_size = kernel_size // 2
        self.learnable_sigmas = nn.ParameterList([
            nn.Parameter(torch.tensor([1.0], dtype=torch.float32))
            for _ in range(num_filters)
        ])

    def forward(self, x):
        gaussian_kernels = [
            self._build_gaussian_kernel(
                kernel_size=self.kernel_size, sigma=sigma
            ).repeat(self.num_channels, 1, 1, 1)
            for sigma in self.learnable_sigmas
        ]

        smoothed_features = [
            F.conv2d(
                F.pad(
                    x,
                    (self.padding_size, self.padding_size,
                     self.padding_size, self.padding_size),
                    mode='replicate'),
                weight=kernel.to(x.device),
                groups=self.num_channels)
            for kernel in gaussian_kernels
        ]
        return torch.cat(smoothed_features, dim=1)

    def _build_gaussian_kernel(self, kernel_size, sigma):
        gaussian_kernel = torch.zeros(
            1, 1, kernel_size, kernel_size,
            dtype=sigma.dtype, device=sigma.device)
        kernel_center = kernel_size // 2
        for row_index in range(kernel_size):
            for col_index in range(kernel_size):
                gaussian_kernel[:, :, row_index, col_index] = torch.exp(
                    -((row_index - kernel_center) ** 2 +
                      (col_index - kernel_center) ** 2) /
                    (2 * sigma ** 2))
        return gaussian_kernel / gaussian_kernel.sum()


class LowFrequencyGuidedFeaturePurification(nn.Module):
    def __init__(self,
                 in_channels,
                 wavelet_type='haar',
                 padding_mode='symmetric',
                 enable_gaussian_smoothing=True,
                 high_frequency_threshold=0.5):
        super().__init__()
        self.wavelet_decomposition = DiscreteWaveletTransform2D(
            wavelet_type=wavelet_type, padding_mode=padding_mode)
        self.wavelet_reconstruction = InverseDiscreteWaveletTransform2D(
            wavelet_type=wavelet_type, padding_mode=padding_mode)
        self.enable_gaussian_smoothing = enable_gaussian_smoothing
        self.high_frequency_threshold = high_frequency_threshold
        self.low_frequency_spatial_attention = SpatialAttentionMapGenerator()

        if self.enable_gaussian_smoothing:
            self.high_frequency_gaussian_smoothing = \
                LearnableGaussianSmoothingBank(
                    kernel_size=3,
                    num_filters=1,
                    num_channels=3 * in_channels)

    def forward(self, x):
        _, channels, _, _ = x.shape
        wavelet_feature = self.wavelet_decomposition(x)
        low_frequency_feature = wavelet_feature[:, :channels, :, :]
        high_frequency_feature = wavelet_feature[:, channels:, :, :]

        target_spatial_weight_map = self.low_frequency_spatial_attention(
            low_frequency_feature)
        modulated_high_frequency_feature = (
            high_frequency_feature * target_spatial_weight_map)

        if self.enable_gaussian_smoothing:
            smoothed_high_frequency_feature = (
                self.high_frequency_gaussian_smoothing(
                    modulated_high_frequency_feature))
            low_confidence_high_frequency_mask = (
                modulated_high_frequency_feature.abs() <
                self.high_frequency_threshold).float()
            purified_high_frequency_feature = (
                modulated_high_frequency_feature *
                (1.0 - low_confidence_high_frequency_mask) +
                smoothed_high_frequency_feature *
                low_confidence_high_frequency_mask)
        else:
            purified_high_frequency_feature = modulated_high_frequency_feature

        return self.wavelet_reconstruction(
            low_frequency_feature, purified_high_frequency_feature)


class AdaptiveCombiner(nn.Module):
    def __init__(self):
        super().__init__()
        self.weight = nn.Parameter(torch.randn(1, 1, 1, 1))

    def forward(self, detail_feat, main_feat):
        gate = torch.sigmoid(self.weight)
        return gate * detail_feat + (1 - gate) * main_feat


class DetailPreservingContextFusion(nn.Module):
    def __init__(self,
                 channels,
                 out_channels,
                 conv_cfg=None,
                 norm_cfg=None,
                 act_cfg=dict(type='ReLU'),
                 group_splits=4,
                 align_corners=False):
        super().__init__()
        assert channels % group_splits == 0, \
            'DPCF-style fusion expects channels divisible by group_splits.'
        self.group_splits = group_splits
        self.align_corners = align_corners
        self.combiner = AdaptiveCombiner()
        self.tail_conv = ConvModule(
            channels,
            out_channels,
            1,
            conv_cfg=conv_cfg,
            norm_cfg=norm_cfg,
            act_cfg=act_cfg)

    def forward(self, detail_feat, main_feat):
        if main_feat.shape[2:] != detail_feat.shape[2:]:
            main_feat = resize(
                main_feat,
                size=detail_feat.shape[2:],
                mode='bilinear',
                align_corners=self.align_corners)

        detail_chunks = torch.chunk(detail_feat, self.group_splits, dim=1)
        main_chunks = torch.chunk(main_feat, self.group_splits, dim=1)
        fused_chunks = [
            self.combiner(detail_chunk, main_chunk)
            for detail_chunk, main_chunk in zip(detail_chunks, main_chunks)
        ]
        fused_feat = torch.cat(fused_chunks, dim=1)
        return self.tail_conv(fused_feat)


class BoundaryAwareAdaptiveCombiner(AdaptiveCombiner):
    def __init__(self,
                 channels,
                 use_boundary=True,
                 reduction=4,
                 align_corners=False):
        super().__init__()
        self.use_boundary = use_boundary
        self.align_corners = align_corners
        hidden_channels = max(channels // reduction, 8)
        gate_in_channels = channels * 2 + (1 if use_boundary else 0)

        self.spatial_gate = nn.Sequential(
            nn.Conv2d(gate_in_channels, hidden_channels, 1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(
                hidden_channels,
                hidden_channels,
                3,
                padding=1,
                groups=hidden_channels,
                bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden_channels, channels, 1, bias=True),
            nn.Sigmoid())

        self.channel_gate = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(gate_in_channels, hidden_channels, 1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden_channels, channels, 1, bias=True),
            nn.Sigmoid())

    def forward(self, detail_feat, main_feat, boundary_logit=None):
        gate_inputs = [detail_feat, main_feat]
        if self.use_boundary and boundary_logit is not None:
            if boundary_logit.shape[2:] != detail_feat.shape[2:]:
                boundary_logit = resize(
                    boundary_logit,
                    size=detail_feat.shape[2:],
                    mode='bilinear',
                    align_corners=self.align_corners)
            gate_inputs.append(boundary_logit.sigmoid())

        gate_feat = torch.cat(gate_inputs, dim=1)
        spatial_gate = self.spatial_gate(gate_feat)
        channel_gate = self.channel_gate(gate_feat)
        gate = 0.5 * (spatial_gate + channel_gate)
        fused_feat = gate * detail_feat + (1 - gate) * main_feat
        return fused_feat, gate


class BoundaryAwareDetailPreservingContextFusion(
        DetailPreservingContextFusion):
    def __init__(self,
                 channels,
                 out_channels,
                 conv_cfg=None,
                 norm_cfg=None,
                 act_cfg=dict(type='ReLU'),
                 group_splits=4,
                 use_boundary=True,
                 align_corners=False):
        super().__init__(
            channels=channels,
            out_channels=out_channels,
            conv_cfg=conv_cfg,
            norm_cfg=norm_cfg,
            act_cfg=act_cfg,
            group_splits=group_splits,
            align_corners=align_corners)
        chunk_channels = channels // group_splits
        self.combiners = nn.ModuleList([
            BoundaryAwareAdaptiveCombiner(
                channels=chunk_channels,
                use_boundary=use_boundary,
                align_corners=align_corners)
            for _ in range(group_splits)
        ])
        self.tail_conv = nn.Sequential(
            ConvModule(
                channels,
                channels,
                3,
                padding=1,
                conv_cfg=conv_cfg,
                norm_cfg=norm_cfg,
                act_cfg=act_cfg),
            ConvModule(
                channels,
                out_channels,
                1,
                conv_cfg=conv_cfg,
                norm_cfg=norm_cfg,
                act_cfg=act_cfg))

    def forward(self, detail_feat, main_feat, boundary_logit=None):
        if main_feat.shape[2:] != detail_feat.shape[2:]:
            main_feat = resize(
                main_feat,
                size=detail_feat.shape[2:],
                mode='bilinear',
                align_corners=self.align_corners)

        detail_chunks = torch.chunk(detail_feat, self.group_splits, dim=1)
        main_chunks = torch.chunk(main_feat, self.group_splits, dim=1)
        fused_chunks = [
            combiner(detail_chunk, main_chunk, boundary_logit)[0]
            for combiner, detail_chunk, main_chunk in zip(
                self.combiners, detail_chunks, main_chunks)
        ]
        fused_feat = torch.cat(fused_chunks, dim=1)
        return self.tail_conv(fused_feat) + detail_feat


class IIAInspiredDetailGate(nn.Module):
    """A light gate inspired by information integration attention.

    It jointly uses stage1 details, stage2 semantics and their disagreement
    map to estimate a refined keep-mask for nighttime detail filtering.
    """

    def __init__(self, channels, reduction=4):
        super().__init__()
        hidden_channels = max(channels // reduction, 8)
        self.pre_fuse = nn.Sequential(
            nn.Conv2d(channels * 3, channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(channels),
            nn.ReLU(inplace=True))
        self.local_refine = nn.Sequential(
            nn.Conv2d(
                channels,
                channels,
                kernel_size=3,
                padding=1,
                groups=channels,
                bias=False),
            nn.BatchNorm2d(channels),
            nn.ReLU(inplace=True))
        self.semantic_gate = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=1, bias=True),
            nn.Sigmoid())
        self.channel_gate = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(channels, hidden_channels, kernel_size=1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden_channels, channels, kernel_size=1, bias=True),
            nn.Sigmoid())
        self.spatial_gate = nn.Sequential(
            nn.Conv2d(4, hidden_channels, kernel_size=3, padding=1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden_channels, 1, kernel_size=1, bias=True),
            nn.Sigmoid())

    def forward(self, stage1_feat, stage2_feat):
        disagreement = torch.abs(stage1_feat - stage2_feat)
        fused_feat = self.pre_fuse(
            torch.cat([stage1_feat, stage2_feat, disagreement], dim=1))
        fused_feat = fused_feat + self.local_refine(fused_feat)

        semantic_gate = self.semantic_gate(stage2_feat)
        channel_gate = self.channel_gate(fused_feat)
        spatial_gate = self.spatial_gate(torch.cat([
            torch.mean(fused_feat, dim=1, keepdim=True),
            torch.max(fused_feat, dim=1, keepdim=True)[0],
            torch.mean(disagreement, dim=1, keepdim=True),
            torch.max(disagreement, dim=1, keepdim=True)[0]
        ], dim=1))

        gate = semantic_gate
        gate = gate * (0.5 + 0.5 * channel_gate)
        gate = gate * (0.5 + 0.5 * spatial_gate)
        return gate


@HEADS.register_module()
class LightHamHead(BaseDecodeHead):
    """Is Attention Better Than Matrix Decomposition?
    This head is the implementation of `HamNet
    <https://arxiv.org/abs/2109.04553>`_.
    Args:
        ham_channels (int): input channels for Hamburger.
        ham_kwargs (int): kwagrs for Ham.

    TODO: 
        Add other MD models (Ham). 
    """

    def __init__(self,
                 ham_channels=512,
                 ham_kwargs=dict(),
                 **kwargs):
        super(LightHamHead, self).__init__(
            input_transform='multiple_select', **kwargs)
        self.ham_channels = ham_channels

        self.squeeze = ConvModule(
            sum(self.in_channels),
            self.ham_channels,
            1,
            conv_cfg=self.conv_cfg,
            norm_cfg=self.norm_cfg,
            act_cfg=self.act_cfg)

        self.hamburger = Hamburger(ham_channels, ham_kwargs, **kwargs)

        self.align = ConvModule(
            self.ham_channels,
            self.channels,
            1,
            conv_cfg=self.conv_cfg,
            norm_cfg=self.norm_cfg,
            act_cfg=self.act_cfg)

    def forward(self, inputs):
        """Forward function."""
        inputs = self._transform_inputs(inputs)

        inputs = [resize(
            level,
            size=inputs[0].shape[2:],
            mode='bilinear',
            align_corners=self.align_corners
        ) for level in inputs]

        inputs = torch.cat(inputs, dim=1)
        x = self.squeeze(inputs)

        x = self.hamburger(x)

        output = self.align(x)
        output = self.cls_seg(output)
        return output


from .FreqFusion import FreqFusion
@HEADS.register_module()
class LightHamHeadFreqAware(LightHamHead):
    """Is Attention Better Than Matrix Decomposition?
    This head is the implementation of `HamNet
    <https://arxiv.org/abs/2109.04553>`_.
    Args:
        ham_channels (int): input channels for Hamburger.
        ham_kwargs (int): kwagrs for Ham.

    TODO: 
        Add other MD models (Ham). 
    """

    def __init__(self,
                use_high_pass=True, 
                use_low_pass=True,
                compress_ratio=8,
                semi_conv=True,
                low2high_residual=False,
                high2low_residual=False,
                lowpass_kernel=5,
                highpass_kernel=3,
                hamming_window=False,
                feature_resample=True,
                feature_resample_group=4,
                comp_feat_upsample=True,
                use_checkpoint=False,
                feature_resample_norm=True,
                **kwargs):
        super().__init__(**kwargs)
        self.freqfusions = nn.ModuleList()
        in_channels = kwargs.get('in_channels', [])
        self.feature_resample = feature_resample
        self.feature_resample_group = feature_resample_group
        self.use_checkpoint = use_checkpoint
        # from lr to hr
        in_channels = in_channels[::-1]
        pre_c = in_channels[0]
        for c in in_channels[1:]:
            freqfusion = FreqFusion(
                hr_channels=c, lr_channels=pre_c, scale_factor=1, lowpass_kernel=lowpass_kernel, highpass_kernel=highpass_kernel, up_group=1, 
                upsample_mode='nearest', align_corners=False, 
                feature_resample=feature_resample, feature_resample_group=feature_resample_group,
                comp_feat_upsample=comp_feat_upsample,
                hr_residual=True, 
                hamming_window=hamming_window,
                compressed_channels= (pre_c + c) // compress_ratio,
                use_high_pass=use_high_pass, use_low_pass=use_low_pass, semi_conv=semi_conv, 
                feature_resample_norm=feature_resample_norm,
                )                
            self.freqfusions.append(freqfusion)
            pre_c += c

        # from lr to hr
        assert not (low2high_residual and high2low_residual)
        self.low2high_residual = low2high_residual
        self.high2low_residual = high2low_residual
        if low2high_residual:
            self.low2high_convs = nn.ModuleList()
            pre_c = in_channels[0]
            for c in in_channels[1:]:
                self.low2high_convs.append(nn.Conv2d(pre_c, c, 1))
                pre_c = c
        elif high2low_residual:
            self.high2low_convs = nn.ModuleList()
            pre_c = in_channels[0]
            for c in in_channels[1:]:
                self.high2low_convs.append(nn.Conv2d(c, pre_c, 1))
                pre_c += c

    def _forward_feature(self, inputs):
        """Forward function."""
        inputs = self._transform_inputs(inputs)
        # inputs = [resize(
        #     level,
        #     size=inputs[0].shape[2:],
        #     mode='bilinear',
        #     align_corners=self.align_corners
        # ) for level in inputs]

        # from low res to high res
        inputs = inputs[::-1]
        in_channels = self.in_channels[::-1]
        lowres_feat = inputs[0]
        if self.low2high_residual:
            for pre_c, hires_feat, freqfusion, low2high_conv in zip(in_channels[:-1], inputs[1:], self.freqfusions, self.low2high_convs):
                _, hires_feat, lowres_feat = freqfusion(hr_feat=hires_feat, lr_feat=lowres_feat, use_checkpoint=self.use_checkpoint)
                lowres_feat = torch.cat([hires_feat + low2high_conv(lowres_feat[:, :pre_c]), lowres_feat], dim=1)
            pass
        else:
            for idx, (hires_feat, freqfusion) in enumerate(zip(inputs[1:], self.freqfusions)):
                _, hires_feat, lowres_feat = freqfusion(hr_feat=hires_feat, lr_feat=lowres_feat, use_checkpoint=self.use_checkpoint)
                if self.feature_resample:
                    b, _, h, w = hires_feat.shape
                    lowres_feat = torch.cat([hires_feat.reshape(b * self.feature_resample_group, -1, h, w), 
                                             lowres_feat.reshape(b * self.feature_resample_group, -1, h, w)], dim=1).reshape(b, -1, h, w)
                else:
                    lowres_feat = torch.cat([hires_feat, lowres_feat], dim=1)

        # inputs = torch.cat(inputs, dim=1)
        inputs = lowres_feat
        x = self.squeeze(inputs)
        x = self.hamburger(x)
        output = self.align(x)

        # output = self.cls_seg(output)
        return output

    def forward(self, inputs):
        output = self._forward_feature(inputs)
        output = self.cls_seg(output)
        return output


@HEADS.register_module()
class LightHamHeadFreqAwareWithDetail(LightHamHeadFreqAware):
    def __init__(self,
                 use_high_pass=True,
                 use_low_pass=True,
                 compress_ratio=8,
                 semi_conv=True,
                 low2high_residual=False,
                 high2low_residual=False,
                 lowpass_kernel=5,
                 highpass_kernel=3,
                 hamming_window=False,
                 feature_resample=True,
                 feature_resample_group=4,
                 comp_feat_upsample=True,
                 use_checkpoint=False,
                 feature_resample_norm=True,
                 detail_channels=64,
                 detail_num_convs=2,
                 **kwargs):
        super().__init__(
            use_high_pass=use_high_pass,
            use_low_pass=use_low_pass,
            compress_ratio=compress_ratio,
            semi_conv=semi_conv,
            low2high_residual=low2high_residual,
            high2low_residual=high2low_residual,
            lowpass_kernel=lowpass_kernel,
            highpass_kernel=highpass_kernel,
            hamming_window=hamming_window,
            feature_resample=feature_resample,
            feature_resample_group=feature_resample_group,
            comp_feat_upsample=comp_feat_upsample,
            use_checkpoint=use_checkpoint,
            feature_resample_norm=feature_resample_norm,
            **kwargs)

        assert len(self.in_channels) >= 4, \
            'LightHamHeadFreqAwareWithDetail expects stage1-4 features.'

        self.detail_in_channels = detail_channels * 2
        self.main_in_channels = self.in_channels[1:]

        # Rebuild the freq-aware path for stage2/3/4 so pretrained head weights
        # still map to the original module names and shapes.
        self.freqfusions = nn.ModuleList()
        main_in_channels = self.main_in_channels[::-1]
        pre_c = main_in_channels[0]
        for c in main_in_channels[1:]:
            freqfusion = FreqFusion(
                hr_channels=c,
                lr_channels=pre_c,
                scale_factor=1,
                lowpass_kernel=lowpass_kernel,
                highpass_kernel=highpass_kernel,
                up_group=1,
                upsample_mode='nearest',
                align_corners=False,
                feature_resample=feature_resample,
                feature_resample_group=feature_resample_group,
                comp_feat_upsample=comp_feat_upsample,
                hr_residual=True,
                hamming_window=hamming_window,
                compressed_channels=(pre_c + c) // compress_ratio,
                use_high_pass=use_high_pass,
                use_low_pass=use_low_pass,
                semi_conv=semi_conv,
                feature_resample_norm=feature_resample_norm,
            )
            self.freqfusions.append(freqfusion)
            pre_c += c

        self.low2high_residual = low2high_residual
        self.high2low_residual = high2low_residual
        if low2high_residual:
            self.low2high_convs = nn.ModuleList()
            pre_c = main_in_channels[0]
            for c in main_in_channels[1:]:
                self.low2high_convs.append(nn.Conv2d(pre_c, c, 1))
                pre_c = c
        elif high2low_residual:
            self.high2low_convs = nn.ModuleList()
            pre_c = main_in_channels[0]
            for c in main_in_channels[1:]:
                self.high2low_convs.append(nn.Conv2d(c, pre_c, 1))
                pre_c += c

        self.squeeze = ConvModule(
            sum(self.main_in_channels),
            self.ham_channels,
            1,
            conv_cfg=self.conv_cfg,
            norm_cfg=self.norm_cfg,
            act_cfg=self.act_cfg)




        self.detail_stage1_proj = ConvModule(
            self.in_channels[0],
            detail_channels,
            1,
            conv_cfg=self.conv_cfg,
            norm_cfg=self.norm_cfg,
            act_cfg=self.act_cfg)
        self.detail_stage2_proj = ConvModule(
            self.in_channels[1],
            detail_channels,
            1,
            conv_cfg=self.conv_cfg,
            norm_cfg=self.norm_cfg,
            act_cfg=self.act_cfg)
        self.detail_gate = nn.Sequential(
            nn.Conv2d(detail_channels, detail_channels, kernel_size=1),
            nn.Sigmoid())

        detail_layers = []
        in_channels = self.detail_in_channels
        for _ in range(detail_num_convs):
            detail_layers.append(
                ConvModule(
                    in_channels,
                    detail_channels,
                    3,
                    padding=1,
                    conv_cfg=self.conv_cfg,
                    norm_cfg=self.norm_cfg,
                    act_cfg=self.act_cfg))
            in_channels = detail_channels
        self.detail_proj = nn.Sequential(*detail_layers)
        self.detail_mhcb = MultiScaleHybridCrossBlock(detail_channels, detail_channels)

        self.detail_fuse = ConvModule(
            self.channels + detail_channels,
            self.channels,
            3,
            padding=1,
            conv_cfg=self.conv_cfg,
            norm_cfg=self.norm_cfg,
            act_cfg=self.act_cfg)






    def _forward_main_feature(self, inputs):
        inputs = inputs[::-1]
        in_channels = self.main_in_channels[::-1]
        lowres_feat = inputs[0]
        if self.low2high_residual:
            for pre_c, hires_feat, freqfusion, low2high_conv in zip(
                    in_channels[:-1], inputs[1:], self.freqfusions,
                    self.low2high_convs):
                _, hires_feat, lowres_feat = freqfusion(
                    hr_feat=hires_feat,
                    lr_feat=lowres_feat,
                    use_checkpoint=self.use_checkpoint)
                lowres_feat = torch.cat(
                    [hires_feat + low2high_conv(lowres_feat[:, :pre_c]),
                     lowres_feat],
                    dim=1)
        else:
            for hires_feat, freqfusion in zip(inputs[1:], self.freqfusions):
                _, hires_feat, lowres_feat = freqfusion(
                    hr_feat=hires_feat,
                    lr_feat=lowres_feat,
                    use_checkpoint=self.use_checkpoint)
                if self.feature_resample:
                    b, _, h, w = hires_feat.shape
                    lowres_feat = torch.cat(
                        [hires_feat.reshape(
                            b * self.feature_resample_group, -1, h, w),
                         lowres_feat.reshape(
                             b * self.feature_resample_group, -1, h, w)],
                        dim=1).reshape(b, -1, h, w)
                else:
                    lowres_feat = torch.cat([hires_feat, lowres_feat], dim=1)

        x = self.squeeze(lowres_feat)
        x = self.hamburger(x)
        output = self.align(x)
        return output

    def forward(self, inputs):
        inputs = self._transform_inputs(inputs)
        stage1_feat = self.detail_stage1_proj(inputs[0])
        stage2_feat = resize(
            inputs[1],
            size=stage1_feat.shape[2:],
            mode='bilinear',
            align_corners=self.align_corners)
        stage2_feat = self.detail_stage2_proj(stage2_feat)
        detail_gate = self.detail_gate(stage2_feat)
        guided_stage1_feat = stage1_feat * detail_gate
        detail_base = self.detail_proj(
            torch.cat([guided_stage1_feat, stage2_feat], dim=1))
        detail_feat = self.detail_mhcb(detail_base)
        main_feat = self._forward_main_feature(inputs[1:])
        main_feat = resize(
            main_feat,
            size=detail_feat.shape[2:],
            mode='bilinear',
            align_corners=self.align_corners)
        output = self.detail_fuse(torch.cat([main_feat, detail_feat], dim=1))
        output = self.cls_seg(output)
        return output


@HEADS.register_module()
class LightHamHeadFreqAwareboundry(LightHamHeadFreqAwareWithDetail):
    def __init__(self,
                 boundary_threshold=0.1,
                 boundary_loss_decode=dict(
                     type='CrossEntropyLoss',
                     use_sigmoid=True,
                     loss_name='loss_boundary',
                     loss_weight=0.2),
                 boundary_guidance=True,
                 boundary_mid_channels=None,
                 detail_channels=64,
                 **kwargs):
        super().__init__(detail_channels=detail_channels, **kwargs)
        self.boundary_threshold = boundary_threshold
        self.boundary_guidance = boundary_guidance
        boundary_mid_channels = (
            detail_channels if boundary_mid_channels is None
            else boundary_mid_channels)
        self.boundary_head = nn.Sequential(
            ConvModule(
                detail_channels,
                boundary_mid_channels,
                3,
                padding=1,
                conv_cfg=self.conv_cfg,
                norm_cfg=self.norm_cfg,
                act_cfg=self.act_cfg),
            nn.Conv2d(boundary_mid_channels, 1, kernel_size=1))

        if isinstance(boundary_loss_decode, dict):
            self.boundary_loss_decode = build_loss(boundary_loss_decode)
        elif isinstance(boundary_loss_decode, (list, tuple)):
            self.boundary_loss_decode = nn.ModuleList()
            for loss in boundary_loss_decode:
                self.boundary_loss_decode.append(build_loss(loss))
        else:
            raise TypeError('boundary_loss_decode must be a dict or '
                            f'sequence of dict, but got '
                            f'{type(boundary_loss_decode)}')

        self.register_buffer(
            'laplacian_kernel',
            torch.tensor([-1, -1, -1, -1, 8, -1, -1, -1, -1],
                         dtype=torch.float32,
                         requires_grad=False).reshape((1, 1, 3, 3)))
        self.register_buffer(
            'fusion_kernel',
            torch.tensor([[0.6], [0.3], [0.1]],
                         dtype=torch.float32,
                         requires_grad=False).reshape((1, 3, 1, 1)))

    def _forward_with_boundary(self, inputs):
        inputs = self._transform_inputs(inputs)
        stage1_feat = self.detail_stage1_proj(inputs[0])
        stage2_feat = resize(
            inputs[1],
            size=stage1_feat.shape[2:],
            mode='bilinear',
            align_corners=self.align_corners)
        stage2_feat = self.detail_stage2_proj(stage2_feat)
        detail_gate = self.detail_gate(stage2_feat)
        guided_stage1_feat = stage1_feat * detail_gate
        detail_base = self.detail_proj(
            torch.cat([guided_stage1_feat, stage2_feat], dim=1))
        detail_feat = self.detail_mhcb(detail_base)
        boundary_logit = self.boundary_head(detail_feat)
        if self.boundary_guidance:
            detail_feat = detail_feat * (1 + boundary_logit.sigmoid())

        main_feat = self._forward_main_feature(inputs[1:])
        main_feat = resize(
            main_feat,
            size=detail_feat.shape[2:],
            mode='bilinear',
            align_corners=self.align_corners)
        output = self.detail_fuse(torch.cat([main_feat, detail_feat], dim=1))
        output = self.cls_seg(output)
        return output, boundary_logit

    def forward(self, inputs):
        output, _ = self._forward_with_boundary(inputs)
        return output

    def forward_train(self, inputs, img_metas, gt_semantic_seg, train_cfg):
        seg_logit, boundary_logit = self._forward_with_boundary(inputs)
        losses = self.losses(seg_logit, gt_semantic_seg)
        losses.update(self.boundary_losses(boundary_logit, gt_semantic_seg))
        return losses

    def _generate_boundary_targets(self, seg_label):
        valid_mask = (seg_label != self.ignore_index).float()
        seg_label = seg_label.float().clone()
        seg_label[valid_mask == 0] = 0

        def _laplacian_boundary(targets, mask, stride=1):
            boundary = F.conv2d(
                targets, self.laplacian_kernel, stride=stride, padding=1)
            boundary = boundary.clamp(min=0)
            pooled_mask = F.avg_pool2d(
                mask, kernel_size=3, stride=stride, padding=1)
            boundary[pooled_mask < 1] = 0
            return boundary

        boundary_targets = _laplacian_boundary(seg_label, valid_mask)
        boundary_targets = (boundary_targets > self.boundary_threshold).float()

        boundary_targets_x2 = _laplacian_boundary(
            seg_label, valid_mask, stride=2)
        boundary_targets_x4 = _laplacian_boundary(
            seg_label, valid_mask, stride=4)

        boundary_targets_x2 = F.interpolate(
            boundary_targets_x2,
            boundary_targets.shape[2:],
            mode='nearest')
        boundary_targets_x4 = F.interpolate(
            boundary_targets_x4,
            boundary_targets.shape[2:],
            mode='nearest')

        boundary_targets_x2 = (
            boundary_targets_x2 > self.boundary_threshold).float()
        boundary_targets_x4 = (
            boundary_targets_x4 > self.boundary_threshold).float()

        boundary_targets = torch.stack(
            (boundary_targets, boundary_targets_x2, boundary_targets_x4),
            dim=1).squeeze(2)
        boundary_targets = F.conv2d(boundary_targets, self.fusion_kernel)
        boundary_targets = (
            boundary_targets > self.boundary_threshold).float()
        boundary_targets[valid_mask == 0] = self.ignore_index
        return boundary_targets

    def boundary_losses(self, boundary_logit, seg_label):
        loss = dict()
        boundary_label = self._generate_boundary_targets(seg_label)
        boundary_logit = resize(
            input=boundary_logit,
            size=boundary_label.shape[2:],
            mode='bilinear',
            align_corners=self.align_corners)
        boundary_label = boundary_label.squeeze(1).long()

        if not isinstance(self.boundary_loss_decode, nn.ModuleList):
            losses_decode = [self.boundary_loss_decode]
        else:
            losses_decode = self.boundary_loss_decode

        for loss_decode in losses_decode:
            if loss_decode.loss_name not in loss:
                loss[loss_decode.loss_name] = loss_decode(
                    boundary_logit,
                    boundary_label,
                    ignore_index=self.ignore_index)
            else:
                loss[loss_decode.loss_name] += loss_decode(
                    boundary_logit,
                    boundary_label,
                    ignore_index=self.ignore_index)

        valid_mask = boundary_label != self.ignore_index
        if valid_mask.any():
            boundary_pred = (boundary_logit.sigmoid().squeeze(1) > 0.5).long()
            loss['acc_boundary'] = (
                boundary_pred.eq(boundary_label)[valid_mask].float().mean() *
                100.0)
        else:
            loss['acc_boundary'] = boundary_logit.new_tensor(0.)
        return loss
@HEADS.register_module()
class LightHamHeadFreqAwareboundryDPCF(LightHamHeadFreqAwareboundry):
    def __init__(self, detail_channels=64, **kwargs):
        super().__init__(detail_channels=detail_channels, **kwargs)
        self.detail_to_main = ConvModule(
            detail_channels,
            self.channels,
            1,
            conv_cfg=self.conv_cfg,
            norm_cfg=self.norm_cfg,
            act_cfg=self.act_cfg)
        self.detail_fuse = DetailPreservingContextFusion(
            channels=self.channels,
            out_channels=self.channels,
            conv_cfg=self.conv_cfg,
            norm_cfg=self.norm_cfg,
            act_cfg=self.act_cfg)

    def _forward_with_boundary(self, inputs):
        inputs = self._transform_inputs(inputs)
        stage1_feat = self.detail_stage1_proj(inputs[0])
        stage2_feat = resize(
            inputs[1],
            size=stage1_feat.shape[2:],
            mode='bilinear',
            align_corners=self.align_corners)
        stage2_feat = self.detail_stage2_proj(stage2_feat)
        detail_gate = self.detail_gate(stage2_feat)
        guided_stage1_feat = stage1_feat * detail_gate
        detail_base = self.detail_proj(
            torch.cat([guided_stage1_feat, stage2_feat], dim=1))
        detail_feat = self.detail_mhcb(detail_base)
        boundary_logit = self.boundary_head(detail_feat)
        if self.boundary_guidance:
            detail_feat = detail_feat * (1 + boundary_logit.sigmoid())

        main_feat = self._forward_main_feature(inputs[1:])
        main_feat = resize(
            main_feat,
            size=detail_feat.shape[2:],
            mode='bilinear',
            align_corners=self.align_corners)
        detail_feat = self.detail_to_main(detail_feat)
        output = self.detail_fuse(detail_feat, main_feat)
        output = self.cls_seg(output)
        return output, boundary_logit




@HEADS.register_module()
class LightHamHeadFreqAwareboundryDPCFPFESA(
        LightHamHeadFreqAwareboundryDPCF):
    def __init__(self,
                 detail_channels=64,
                 pfesa_base_ratio=0.1,
                 **kwargs):
        super().__init__(detail_channels=detail_channels, **kwargs)
        self.detail_pfesa = PFESA(base_ratio=pfesa_base_ratio)

    def _forward_with_boundary(self, inputs):
        inputs = self._transform_inputs(inputs)
        stage1_feat = self.detail_stage1_proj(inputs[0])
        stage2_feat = resize(
            inputs[1],
            size=stage1_feat.shape[2:],
            mode='bilinear',
            align_corners=self.align_corners)
        stage2_feat = self.detail_stage2_proj(stage2_feat)
        detail_gate = self.detail_gate(stage2_feat)
        guided_stage1_feat = stage1_feat * detail_gate
        detail_base = self.detail_proj(
            torch.cat([guided_stage1_feat, stage2_feat], dim=1))
        detail_base = self.detail_pfesa(detail_base)
        detail_feat = self.detail_mhcb(detail_base)
        boundary_logit = self.boundary_head(detail_feat)
        if self.boundary_guidance:
            detail_feat = detail_feat * (1 + boundary_logit.sigmoid())

        main_feat = self._forward_main_feature(inputs[1:])
        main_feat = resize(
            main_feat,
            size=detail_feat.shape[2:],
            mode='bilinear',
            align_corners=self.align_corners)
        detail_feat = self.detail_to_main(detail_feat)
        output = self.detail_fuse(detail_feat, main_feat)
        output = self.cls_seg(output)
        return output, boundary_logit




@HEADS.register_module()
class LightHamHeadFreqAwareboundryDPCF222(LightHamHeadFreqAwareboundry):
    def __init__(self, detail_channels=64, dpcf_use_boundary=True, **kwargs):
        super().__init__(detail_channels=detail_channels, **kwargs)
        self.detail_to_main = ConvModule(
            detail_channels,
            self.channels,
            1,
            conv_cfg=self.conv_cfg,
            norm_cfg=self.norm_cfg,
            act_cfg=self.act_cfg)
        self.detail_fuse = BoundaryAwareDetailPreservingContextFusion(
            channels=self.channels,
            out_channels=self.channels,
            conv_cfg=self.conv_cfg,
            norm_cfg=self.norm_cfg,
            act_cfg=self.act_cfg,

            
            group_splits=4,
            use_boundary=dpcf_use_boundary,
            align_corners=self.align_corners)

    def _forward_with_boundary(self, inputs):
        inputs = self._transform_inputs(inputs)
        stage1_feat = self.detail_stage1_proj(inputs[0])
        stage2_feat = resize(
            inputs[1],
            size=stage1_feat.shape[2:],
            mode='bilinear',
            align_corners=self.align_corners)
        stage2_feat = self.detail_stage2_proj(stage2_feat)
        detail_gate = self.detail_gate(stage2_feat)
        guided_stage1_feat = stage1_feat * detail_gate
        detail_base = self.detail_proj(
            torch.cat([guided_stage1_feat, stage2_feat], dim=1))
        detail_feat = self.detail_mhcb(detail_base)
        boundary_logit = self.boundary_head(detail_feat)
        if self.boundary_guidance:
            detail_feat = detail_feat * (1 + boundary_logit.sigmoid())

        main_feat = self._forward_main_feature(inputs[1:])
        main_feat = resize(
            main_feat,
            size=detail_feat.shape[2:],
            mode='bilinear',
            align_corners=self.align_corners)
        detail_feat = self.detail_to_main(detail_feat)
        output = self.detail_fuse(detail_feat, main_feat, boundary_logit)
        output = self.cls_seg(output)
        return output, boundary_logit


@HEADS.register_module()
class LightHamHeadFreqAwareboundryDPCF222PFESA(
        LightHamHeadFreqAwareboundryDPCF222):
    def __init__(self,
                 detail_channels=64,
                 pfesa_base_ratio=0.1,
                 **kwargs):
        super().__init__(detail_channels=detail_channels, **kwargs)
        self.detail_pfesa = PFESA(base_ratio=pfesa_base_ratio)

    def _forward_with_boundary(self, inputs):
        inputs = self._transform_inputs(inputs)
        stage1_feat = self.detail_stage1_proj(inputs[0])
        stage2_feat = resize(
            inputs[1],
            size=stage1_feat.shape[2:],
            mode='bilinear',
            align_corners=self.align_corners)
        stage2_feat = self.detail_stage2_proj(stage2_feat)
        detail_gate = self.detail_gate(stage2_feat)
        guided_stage1_feat = stage1_feat * detail_gate
        detail_base = self.detail_proj(
            torch.cat([guided_stage1_feat, stage2_feat], dim=1))
        detail_base = self.detail_pfesa(detail_base)
        detail_feat = self.detail_mhcb(detail_base)
        boundary_logit = self.boundary_head(detail_feat)
        if self.boundary_guidance:
            detail_feat = detail_feat * (1 + boundary_logit.sigmoid())

        main_feat = self._forward_main_feature(inputs[1:])
        main_feat = resize(
            main_feat,
            size=detail_feat.shape[2:],
            mode='bilinear',
            align_corners=self.align_corners)
        detail_feat = self.detail_to_main(detail_feat)
        output = self.detail_fuse(detail_feat, main_feat, boundary_logit)
        output = self.cls_seg(output)
        return output, boundary_logit


@HEADS.register_module()
class LightHamHeadFreqAwareboundryDPCF222PFESANoBG(
        LightHamHeadFreqAwareboundryDPCF222PFESA):
    """PFESA + boundary-aware DPCF without boundary_guidance.

    Boundary predictions are still supervised and still used by the final
    BoundaryAwareDetailPreservingContextFusion, but they no longer
    multiplicatively amplify detail features inside the detail branch.
    """

    def __init__(self, detail_channels=64, **kwargs):
        super().__init__(detail_channels=detail_channels, **kwargs)
        self.boundary_guidance = False


@HEADS.register_module()
class LightHamHeadFreqAwareboundryDPCF222PFESAstronggate(
        LightHamHeadFreqAwareboundryDPCF222PFESA):
    def __init__(self,
                 detail_channels=64,
                 strong_gate_reduction=4,
                 **kwargs):
        super().__init__(detail_channels=detail_channels, **kwargs)
        self.detail_gate = NightSceneStrongDetailGate(
            channels=detail_channels,
            reduction=strong_gate_reduction)

    def _forward_with_boundary(self, inputs):
        inputs = self._transform_inputs(inputs)
        stage1_feat = self.detail_stage1_proj(inputs[0])
        stage2_feat = resize(
            inputs[1],
            size=stage1_feat.shape[2:],
            mode='bilinear',
            align_corners=self.align_corners)
        stage2_feat = self.detail_stage2_proj(stage2_feat)
        detail_gate = self.detail_gate(stage1_feat, stage2_feat)
        guided_stage1_feat = stage1_feat * detail_gate
        detail_base = self.detail_proj(
            torch.cat([guided_stage1_feat, stage2_feat], dim=1))
        detail_base = self.detail_pfesa(detail_base)
        detail_feat = self.detail_mhcb(detail_base)
        boundary_logit = self.boundary_head(detail_feat)
        if self.boundary_guidance:
            detail_feat = detail_feat * (1 + boundary_logit.sigmoid())

        main_feat = self._forward_main_feature(inputs[1:])
        main_feat = resize(
            main_feat,
            size=detail_feat.shape[2:],
            mode='bilinear',
            align_corners=self.align_corners)
        detail_feat = self.detail_to_main(detail_feat)
        output = self.detail_fuse(detail_feat, main_feat, boundary_logit)
        output = self.cls_seg(output)
        return output, boundary_logit


