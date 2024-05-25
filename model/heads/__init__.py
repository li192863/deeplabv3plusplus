from .head import get_head
from .aspp import ASPP
from .fcn import FCNHead
from .merge import Merge, get_merge


__all__ = [
    'ASPP',
    'FCNHead',
    'get_head',
    'Merge',
    'get_merge'
]


