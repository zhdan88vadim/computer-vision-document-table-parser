"""
Document Parser Application
"""
from app.config import Config
from app.alignment import ImageAligner
from app.block_detection import BlockDetector
from app.ocr_engine import OCRProcessor
from app.field_mapper import FieldMapper
from app.visualization import Visualizer
from app.testing import Tester
from app.utils import clean_output_folder

__all__ = [
    'Config',
    'ImageAligner',
    'BlockDetector',
    'OCRProcessor',
    'FieldMapper',
    'Visualizer',
    'Tester',
    'clean_output_folder'
]