# trigger registration of the viewer widgets
from .process import WorkChainSelector
from .structures import StructureSelectionStep
from .widgets import CalcJobNodeViewerWidget  # noqa: F401

__all__ = [
    "WorkChainSelector",
    "StructureSelectionStep",
]
