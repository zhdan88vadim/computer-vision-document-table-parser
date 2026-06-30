"""
Configuration management
"""
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional


@dataclass
class Config:
    """Configuration for document parser"""
    
    # Paths
    input_dir: Path = Path("./input_images")
    output_dir: Path = Path("./output_finish")
    debug_dir: Path = Path("./output_finish/debug")
    expected_dir: Path = Path("./test_data/expected")
    test_output_dir: Path = Path("./test_data/output")
    
    # Block detection parameters
    dilation_size: int = 7
    std_multiplier: float = 1.0
    min_size: int = 500
    
    # OCR parameters
    ocr_languages: List[str] = field(default_factory=lambda: ['ru', 'en'])
    use_gpu: bool = False
    ocr_padding: int = 10
    
    # Testing parameters
    min_similarity_score: int = 80
    
    # Field mapping: box_index -> field_name
    field_mapping: Dict[int, str] = field(default_factory=lambda: {
        1: 'Выписка из протокола',
        2: 'дата',
        3: 'номер',
        4: 'пункт',
        5: 'Наименование объекта',
        6: 'Авторы проекта',
        7: 'Генеральная проектная организация',
        8: 'Застройщик',
        9: 'Референт',
        10: 'Докладчик',
        11: 'Выступили'
    })
    
    # Extra fields to add to mapped data
    extra_fields: Dict[str, str] = field(default_factory=lambda: {
        'Рассмотрение на рабочей комиссии': ''
    })
    
    def __post_init__(self) -> None:
        """Create directories if they don't exist"""
        self.input_dir.mkdir(parents=True, exist_ok=True)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.debug_dir.mkdir(parents=True, exist_ok=True)
        self.expected_dir.mkdir(parents=True, exist_ok=True)
        self.test_output_dir.mkdir(parents=True, exist_ok=True)