from .builder import DATASETS, PIPELINES, build_dataloader, build_dataset
from .custom import CustomDataset
from .retinaface import RetinaFaceDataset
from .samplers import DistributedGroupSampler, DistributedSampler, GroupSampler
from .utils import replace_ImageToTensor
from .wider_face import WIDERFaceDataset
from .xml_style import XMLDataset

try:
    from .dataset_wrappers import (ClassBalancedDataset, ConcatDataset,
                                   RepeatDataset)
    from .cityscapes import CityscapesDataset
    from .coco import CocoDataset
    from .deepfashion import DeepFashionDataset
    from .lvis import LVISDataset, LVISV1Dataset, LVISV05Dataset
    from .voc import VOCDataset
except Exception:  # pragma: no cover
    ClassBalancedDataset = ConcatDataset = RepeatDataset = None
    CityscapesDataset = CocoDataset = DeepFashionDataset = None
    LVISDataset = LVISV1Dataset = LVISV05Dataset = VOCDataset = None

__all__ = [
    'CustomDataset', 'XMLDataset', 'CocoDataset', 'DeepFashionDataset',
    'VOCDataset', 'CityscapesDataset', 'LVISDataset', 'LVISV05Dataset',
    'LVISV1Dataset', 'GroupSampler', 'DistributedGroupSampler',
    'DistributedSampler', 'build_dataloader', 'ConcatDataset', 'RepeatDataset',
    'RetinaFaceDataset',
    'ClassBalancedDataset', 'WIDERFaceDataset', 'DATASETS', 'PIPELINES',
    'build_dataset', 'replace_ImageToTensor'
]
