"""Model implementations for teacher and student networks."""

from .teacher import CLIPTeacher
from .student import StudentCNN

__all__ = [
    "CLIPTeacher",
    "StudentCNN",
]
