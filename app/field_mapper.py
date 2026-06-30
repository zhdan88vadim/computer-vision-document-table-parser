"""
Field mapping and data organization
"""
import json
import os
from typing import Dict, List, Any, Optional
from pathlib import Path


class FieldMapper:
    """Maps detected boxes to document fields"""
    
    def __init__(self, field_mapping: Dict[int, str], extra_fields: Optional[Dict[str, str]] = None):
        """
        Initialize field mapper
        
        Args:
            field_mapping: Dictionary mapping box_index to field_name
            extra_fields: Additional fields to add to output
        """
        self.field_mapping = field_mapping
        self.extra_fields = extra_fields or {}
    
    def map_fields(self, ocr_results: List[Dict[str, Any]]) -> Dict[str, str]:
        """
        Map OCR results to field names
        
        Args:
            ocr_results: List of OCR results
            
        Returns:
            Dictionary of field_name -> text
        """
        mapped_data = {}
        
        for result in ocr_results:
            box_index = result['box_index']
            text = result['text']
            
            # If mapping exists for this box
            if box_index in self.field_mapping:
                field_name = self.field_mapping[box_index]
                mapped_data[field_name] = text
            else:
                # If no mapping, use box_index as name
                mapped_data[f"box_{box_index}"] = text
        
        # Add extra fields
        mapped_data.update(self.extra_fields)
        
        return mapped_data
    
    def save_mapped_results(self, img_name: str, mapped_data: Dict[str, str], 
                           output_dir: Path) -> None:
        """
        Save mapped results to JSON
        
        Args:
            img_name: Image name
            mapped_data: Mapped field data
            output_dir: Output directory
        """
        json_path = output_dir / f"{img_name}_mapped_results.json"
        
        with open(json_path, 'w', encoding='utf-8') as f:
            json.dump(mapped_data, f, indent=2, ensure_ascii=False)
        
        print(f"Mapped results saved to: {json_path}")
    
    def update_mapping(self, new_mapping: Dict[int, str]) -> None:
        """Update field mapping"""
        self.field_mapping.update(new_mapping)