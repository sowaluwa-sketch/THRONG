NUM_CLASSES = 2
CLASS_NAMES = ("Other", "Malicious")

from .model import GTDA_VARIANT, THRONG
from .types import GroupRecord

__all__ = ["CLASS_NAMES", "GTDA_VARIANT", "GroupRecord", "NUM_CLASSES", "THRONG"]
