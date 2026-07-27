import os.path as osp
import mmcv

from .builder import DATASETS
from .custom import CustomDataset


@DATASETS.register_module()
class NightCityDataset(CustomDataset):
    CLASSES = (
        'road', 'sidewalk', 'building', 'wall', 'fence', 'pole',
        'traffic light', 'traffic sign', 'vegetation', 'terrain',
        'sky', 'person', 'rider', 'car', 'truck', 'bus',
        'train', 'motorcycle', 'bicycle'
    )

    PALETTE = [
        [128, 64, 128], [244, 35, 232], [70, 70, 70], [102, 102, 156],
        [190, 153, 153], [153, 153, 153], [250, 170, 30], [220, 220, 0],
        [107, 142, 35], [152, 251, 152], [70, 130, 180], [220, 20, 60],
        [255, 0, 0], [0, 0, 142], [0, 0, 70], [0, 60, 100],
        [0, 80, 100], [0, 0, 230], [119, 11, 32]
    ]

    def __init__(self,
                 pipeline,
                 ann_file,
                 data_root=None,
                 test_mode=False,
                 ignore_index=255,
                 classes=None,
                 palette=None):
        self.ann_file = ann_file
        super(NightCityDataset, self).__init__(
            pipeline=pipeline,
            img_dir='',
            img_suffix='',
            ann_dir='',
            seg_map_suffix='',
            split=None,
            data_root=data_root,
            test_mode=test_mode,
            ignore_index=ignore_index,
            reduce_zero_label=False,
            classes=classes,
            palette=palette)

    def load_annotations(self, img_dir, img_suffix, ann_dir, seg_map_suffix, split):
        ann_file = self.ann_file
        if self.data_root is not None and not osp.isabs(ann_file):
            ann_file = osp.join(self.data_root, ann_file)

        img_infos = []
        for line in mmcv.list_from_file(ann_file):
            parts = line.strip().split()
            if len(parts) < 2:
                continue

            img_path, seg_map_path = parts[:2]

            if self.data_root is not None:
                if not osp.isabs(img_path):
                    img_path = osp.join(self.data_root, img_path)
                if not osp.isabs(seg_map_path):
                    seg_map_path = osp.join(self.data_root, seg_map_path)

            img_infos.append(
                dict(
                    filename=img_path,
                    ann=dict(seg_map=seg_map_path)
                )
            )
        return img_infos

    def pre_pipeline(self, results):
        results['seg_fields'] = []
        results['img_prefix'] = None
        results['seg_prefix'] = None
        if self.custom_classes:
            results['label_map'] = self.label_map
    def get_gt_seg_maps(self, efficient_test=False):
        gt_seg_maps = []
        for img_info in self.img_infos:
            seg_map = img_info['ann']['seg_map']
            if efficient_test:
                gt_seg_map = seg_map
            else:
                gt_seg_map = mmcv.imread(
                    seg_map, flag='unchanged', backend='pillow')
            gt_seg_maps.append(gt_seg_map)
        return gt_seg_maps
