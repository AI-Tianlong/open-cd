from mmengine.visualization import LocalVisBackend
from mmseg.models import SegTTAModel
from mmseg.visualization import SegLocalVisualizer

from opencd.visualization import CDLocalVisBackend, CDLocalVisualizer


env_cfg = dict(
    cudnn_benchmark=True,
    # cudnn_benchmark=False,
    mp_cfg=dict(mp_start_method='fork', opencv_num_threads=0),
    dist_cfg=dict(backend='nccl'),
)

vis_backends = [dict(type=CDLocalVisBackend)]
visualizer = dict(
    type=CDLocalVisualizer, 
    vis_backends=vis_backends, 
    name='visualizer', 
    alpha=1.0)

log_processor = dict(by_epoch=False)
log_level = 'INFO'
load_from = None
resume = False

tta_model = dict(type=SegTTAModel)
default_scope = 'opencd'
# default_scope = None
