"""
DEIM: DETR with Improved Matching for Fast Convergence
Copyright (c) 2024 The DEIM Authors. All Rights Reserved.
---------------------------------------------------------------------------------
Modified from RT-DETR (https://github.com/lyuwenyu/RT-DETR)
Copyright(c) 2023 lyuwenyu. All Rights Reserved.
"""


from .criterion import ECCriterion
from .decoder import ECTransformer
from .ecvit import ViTAdapter
from .hybrid_encoder import HybridEncoder
from .matcher import HungarianMatcher
from .modeling import ECDet, ECSeg
from .postprocessor import PostProcessor
