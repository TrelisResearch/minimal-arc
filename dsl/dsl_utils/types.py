"""
ARC DSL Types.

This module defines the types used in the ARC DSL.
"""
from dataclasses import dataclass
from typing import List, Tuple, Optional
import numpy as np


@dataclass
class Grid:
    """Representation of a grid in the ARC DSL."""
    data: np.ndarray
    
    def __post_init__(self):
        """Ensure the data is a numpy array."""
        if not isinstance(self.data, np.ndarray):
            self.data = np.array(self.data, dtype=np.int32)
    
    @property
    def shape(self) -> Tuple[int, int]:
        """Get the shape of the grid."""
        return self.data.shape
    
    def copy(self) -> 'Grid':
        """Create a copy of the grid."""
        return Grid(self.data.copy())
    
    def __eq__(self, other) -> bool:
        """Check if two grids are equal."""
        if not isinstance(other, Grid):
            return False
        return np.array_equal(self.data, other.data)


@dataclass
class Object:
    """Representation of an object in the ARC DSL."""
    grid: Grid
    color: int
    position: Tuple[int, int]  # (row, col) of the top-left corner
    
    @property
    def shape(self) -> Tuple[int, int]:
        """Get the shape of the object."""
        return self.grid.shape


@dataclass
class ObjList:
    """Representation of a list of objects in the ARC DSL."""
    objects: List[Object]
    
    def __len__(self) -> int:
        """Get the number of objects."""
        return len(self.objects)
    
    def __getitem__(self, idx: int) -> Object:
        """Get an object by index."""
        return self.objects[idx]


# Type definitions for type checking
class Type:
    """Base class for types in the ARC DSL."""
    pass


class Grid_T(Type):
    """Type for grids."""
    pass


class ObjList_T(Type):
    """Type for object lists."""
    pass


class Int_T(Type):
    """Type for integers."""
    pass


class Bool_T(Type):
    """Type for booleans."""
    pass
