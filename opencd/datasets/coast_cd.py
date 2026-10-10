# Copyright (c) Open-CD. All rights reserved.
from opencd.registry import DATASETS
from .basecddataset import _BaseCDDataset


@DATASETS.register_module()
class Coast_CD_Dataset(_BaseCDDataset):
    """Coast_CD dataset"""
    METAINFO = dict(
        classes=('unchanged', 'water_to_land', 'land_to_water'),
        palette=[[0, 0, 0], [255, 0, 0], [0, 0, 255]])

    def __init__(self,
                 img_suffix='.tif',
                 seg_map_suffix='.tif',
                 **kwargs) -> None:
        super().__init__(
            img_suffix=img_suffix,
            seg_map_suffix=seg_map_suffix,
            **kwargs)
