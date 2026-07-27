# Copyright (c) OpenMMLab. All rights reserved.
"""Batch visualization for MMSegmentation 0.x projects.

This script is intentionally independent from ``tools/test.py``.  In
particular, it never joins ``show_dir`` with an absolute ``ori_filename``;
custom datasets that store full image paths therefore cannot accidentally
write outside the requested output directory.
"""

import argparse
import os.path as osp
import warnings

import mmcv
import numpy as np
import torch
from mmcv.runner import load_checkpoint, wrap_fp16_model
from mmcv.utils import DictAction

try:
    from mmcv.cnn.utils import revert_sync_batchnorm
except ImportError:
    def revert_sync_batchnorm(module):
        """Compatibility fallback for MMCV versions without this helper."""
        output = module
        if isinstance(module, torch.nn.SyncBatchNorm):
            output = torch.nn.BatchNorm2d(
                module.num_features,
                eps=module.eps,
                momentum=module.momentum,
                affine=module.affine,
                track_running_stats=module.track_running_stats)
            if module.affine:
                with torch.no_grad():
                    output.weight.copy_(module.weight)
                    output.bias.copy_(module.bias)
            output.running_mean = module.running_mean
            output.running_var = module.running_var
            output.num_batches_tracked = module.num_batches_tracked
            output.training = module.training

        for name, child in module.named_children():
            output.add_module(name, revert_sync_batchnorm(child))
        return output


from mmseg.datasets import build_dataloader, build_dataset
from mmseg.models import build_segmentor
from mmseg.utils import build_dp, get_device, setup_multi_processes


def parse_args():
    parser = argparse.ArgumentParser(
        description='Batch inference and visualization for a test dataset')
    parser.add_argument('config', help='test config file')
    parser.add_argument('checkpoint', help='model checkpoint')
    parser.add_argument(
        '--show-dir', required=True, help='directory used to save images')
    parser.add_argument(
        '--eval', nargs='+', help='optional metrics, for example: mIoU')
    parser.add_argument(
        '--aug-test', action='store_true',
        help='use 6-scale and flip test-time augmentation, as in test.py')
    parser.add_argument(
        '--opacity', type=float, default=0.5,
        help='opacity of the colored mask in overlay images')
    parser.add_argument(
        '--max-images', type=int, default=None,
        help='only visualize the first N images; evaluate still uses all data')
    parser.add_argument(
        '--no-mask', action='store_true',
        help='do not save the colorized prediction-only images')
    parser.add_argument('--gpu-id', type=int, default=0)
    parser.add_argument(
        '--cfg-options', nargs='+', action=DictAction,
        help='override config values, key=value')
    return parser.parse_args()


def enable_aug_test(test_data_cfg):
    """Enable the same multi-scale flip augmentation as tools/test.py."""
    ratios = [0.5, 0.75, 1.0, 1.25, 1.5, 1.75]

    # Usually test_data_cfg is a dataset config. Handle common wrappers too.
    if isinstance(test_data_cfg, (list, tuple)):
        found = False
        for item in test_data_cfg:
            found = enable_aug_test(item) or found
        return found
    if 'dataset' in test_data_cfg:
        return enable_aug_test(test_data_cfg['dataset'])
    if 'datasets' in test_data_cfg:
        found = False
        for item in test_data_cfg['datasets']:
            found = enable_aug_test(item) or found
        return found

    for transform in test_data_cfg.get('pipeline', []):
        if transform.get('type') == 'MultiScaleFlipAug':
            transform['img_ratios'] = ratios
            transform['flip'] = True
            return True
    return False


def safe_output_name(index, filename):
    """Return a unique flat filename for both Windows and POSIX paths."""
    basename = str(filename).replace('\\', '/').rstrip('/').split('/')[-1]
    stem, _ = osp.splitext(basename)
    stem = stem or 'image'
    return f'{index:06d}_{stem}.png'


def colorize_mask(mask, palette):
    palette = np.asarray(palette, dtype=np.uint8)
    color = np.zeros((*mask.shape, 3), dtype=np.uint8)
    for label, rgb in enumerate(palette):
        color[mask == label] = rgb
    # mmcv.imwrite expects BGR.
    return color[..., ::-1]


def main():
    args = parse_args()
    if not 0 < args.opacity <= 1:
        raise ValueError('--opacity must be in the range (0, 1].')
    if args.max_images is not None and args.max_images < 0:
        raise ValueError('--max-images must be non-negative.')

    cfg = mmcv.Config.fromfile(args.config)
    if args.cfg_options:
        cfg.merge_from_dict(args.cfg_options)
    if args.aug_test and not enable_aug_test(cfg.data.test):
        raise ValueError(
            '--aug-test requires MultiScaleFlipAug in data.test.pipeline.')
    setup_multi_processes(cfg)

    cfg.model.pretrained = None
    cfg.model.train_cfg = None
    cfg.data.test.test_mode = True
    cfg.gpu_ids = [args.gpu_id]

    dataset = build_dataset(cfg.data.test)
    loader_cfg = dict(
        samples_per_gpu=1,
        workers_per_gpu=cfg.data.get('workers_per_gpu', 2),
        num_gpus=1,
        dist=False,
        shuffle=False)
    loader_cfg.update(cfg.data.get('test_dataloader', {}))
    # Visualization below deliberately handles one prediction at a time.
    loader_cfg['samples_per_gpu'] = 1
    loader_cfg['shuffle'] = False
    data_loader = build_dataloader(dataset, **loader_cfg)

    model = build_segmentor(cfg.model, test_cfg=cfg.get('test_cfg'))
    fp16_cfg = cfg.get('fp16')
    if fp16_cfg is not None:
        wrap_fp16_model(model)
    checkpoint = load_checkpoint(model, args.checkpoint, map_location='cpu')
    model.CLASSES = checkpoint.get('meta', {}).get('CLASSES', dataset.CLASSES)
    model.PALETTE = checkpoint.get('meta', {}).get('PALETTE', dataset.PALETTE)

    device = get_device()
    if device == 'cpu':
        warnings.warn('CUDA is unavailable; inference will run on CPU.')
    model = revert_sync_batchnorm(model)
    model = build_dp(model, device, device_ids=cfg.gpu_ids)
    model.eval()

    overlay_dir = osp.abspath(osp.join(args.show_dir, 'overlay'))
    mask_dir = osp.abspath(osp.join(args.show_dir, 'mask'))
    labelmap_dir = osp.abspath(osp.join(args.show_dir, 'labelmap'))  # 新增：单通道 label map
    mmcv.mkdir_or_exist(overlay_dir)
    if not args.no_mask:
        mmcv.mkdir_or_exist(mask_dir)
    mmcv.mkdir_or_exist(labelmap_dir)  # 创建 labelmap 目录

    results = []
    prog_bar = mmcv.ProgressBar(len(dataset))
    visualized = 0
    with torch.no_grad():
        for data in data_loader:
            # rescale=True is important: predictions must match ori_shape.
            result = model(return_loss=False, rescale=True, **data)
            results.extend(result)

            if args.max_images is None or visualized < args.max_images:
                img_meta = data['img_metas'][0].data[0][0]
                filename = img_meta.get('filename') or img_meta.get(
                    'ori_filename')
                if not filename:
                    raise KeyError(
                        'Image metadata has neither filename nor ori_filename')
                image = mmcv.imread(filename)
                output_name = safe_output_name(visualized, filename)
                overlay_file = osp.join(overlay_dir, output_name)

                model.module.show_result(
                    image,
                    result,
                    palette=dataset.PALETTE,
                    show=False,
                    out_file=overlay_file,
                    opacity=args.opacity)

                # 保存单通道 label map（用于后续分析）
                mask = np.asarray(result[0], dtype=np.uint8)
                labelmap_file = osp.join(labelmap_dir, output_name)
                mmcv.imwrite(mask, labelmap_file)

                if not args.no_mask:
                    mmcv.imwrite(
                        colorize_mask(mask, dataset.PALETTE),
                        osp.join(mask_dir, output_name))
                visualized += 1

            for _ in result:
                prog_bar.update()

    print(f'\nSaved {visualized} overlay image(s) to: {overlay_dir}')
    print(f'Saved label map(s) to: {labelmap_dir}')
    if not args.no_mask:
        print(f'Saved color mask image(s) to: {mask_dir}')
    if args.eval:
        metrics = dataset.evaluate(results, metric=args.eval)
        print('Evaluation results:')
        for key, value in metrics.items():
            print(f'  {key}: {value}')


if __name__ == '__main__':
    main()
