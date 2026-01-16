20# dataset settings
from mmcv.transforms.loading import LoadImageFromFile
from mmcv.transforms.processing import (RandomFlip, RandomResize, Resize,
                                        TestTimeAug)
from mmengine.dataset.sampler import DefaultSampler, InfiniteSampler


from mmseg.datasets.transforms.formatting import PackSegInputs
from mmseg.datasets.transforms.loading import (LoadAnnotations,
                                               LoadSingleRSImageFromFile)
from mmseg.datasets.transforms.transforms import (PhotoMetricDistortion,
                                                  RandomCrop)
from mmseg.evaluation import IoUMetric

from mmengine.dataset.sampler import DefaultSampler, InfiniteSampler

from opencd.datasets.transforms.loading import MultiImgLoadImageFromFile, MultiImgLoadAnnotations, MultiImgLoadImageFromFile_gdal
from opencd.datasets.transforms.transforms import MultiImgRandomCrop, MultiImgRandomFlip, MultiImgResize
from opencd.datasets.transforms.formatting import MultiImgPackSegInputs

from opencd.datasets.levir_cd import LEVIR_CD_Dataset
from opencd.datasets.coast_cd import Coast_CD_Dataset

dataset_type = Coast_CD_Dataset
data_root = 'data/CoastCD/3-CoastCD-512'

crop_size = (512, 512)

train_pipeline = [
    dict(type=MultiImgLoadImageFromFile_gdal),
    dict(type=MultiImgLoadAnnotations),
    dict(type=MultiImgRandomCrop, crop_size=crop_size, cat_max_ratio=0.75),
    dict(type=MultiImgRandomFlip, prob=0.5),
    dict(type=MultiImgPackSegInputs)
]

test_pipeline = [
    dict(type=MultiImgLoadImageFromFile_gdal),
    # dict(type=MultiImgResize, scale=(512, 1024), keep_ratio=True),
    # add loading annotation after ``Resize`` because ground truth
    # does not need to do resize data transform
    dict(type=MultiImgLoadAnnotations),
    dict(type=MultiImgPackSegInputs)
]
img_ratios = [0.75, 1.0, 1.25]

tta_pipeline = [
    dict(type='MultiImgLoadImageFromFile_gdal', backend_args=None),
    dict(
        type='TestTimeAug',
        transforms=[
            [
                dict(type='MultiImgResize', scale_factor=r, keep_ratio=True)
                for r in img_ratios
            ],
            [
                dict(type='MultiImgRandomFlip', prob=0., direction='horizontal'),
                dict(type='MultiImgRandomFlip', prob=1., direction='horizontal')
            ],
            [dict(type='MultiImgLoadAnnotations')],
            [dict(type='MultiImgPackSegInputs')]
        ])
]
train_dataloader = dict(
    batch_size=8,
    num_workers=8,
    persistent_workers=True,
    sampler=dict(type=InfiniteSampler, shuffle=True),
    dataset=dict(
        type=dataset_type,
        data_root=data_root,
        data_prefix=dict(
            img_path_from='train/A',
            img_path_to='train/B',
            seg_map_path='train/label'),
        pipeline=train_pipeline))

val_dataloader = dict(
    batch_size=1,
    num_workers=4,
    persistent_workers=True,
    sampler=dict(type=DefaultSampler, shuffle=False),
    dataset=dict(
        type=dataset_type,
        data_root=data_root,
        data_prefix=dict(
            img_path_from='val/A',
            img_path_to='val/B',
            seg_map_path='val/label'),
        pipeline=test_pipeline))

test_dataloader = dict(
    batch_size=1,
    num_workers=4,
    persistent_workers=True,
    sampler=dict(type=DefaultSampler, shuffle=False),
    dataset=dict(
        type=dataset_type,
        data_root=data_root,
        data_prefix=dict(
            img_path_from='val/A',
            img_path_to='val/B',
            seg_map_path='val/label'),
        pipeline=test_pipeline))

val_evaluator = dict(type=IoUMetric, iou_metrics=['mFscore', 'mIoU'])
test_evaluator = val_evaluator