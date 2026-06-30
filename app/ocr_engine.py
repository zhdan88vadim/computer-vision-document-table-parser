"""
OCR processing with EasyOCR
"""
import numpy as np
import json
from typing import List, Dict, Any, Tuple, Optional, Union
from pathlib import Path
import easyocr


class OCRProcessor:
    """Handles OCR text recognition"""
    
    def __init__(self, languages: List[str] = ['ru', 'en'], use_gpu: bool = False):
        """
        Initialize OCR processor
        
        Args:
            languages: List of language codes
            use_gpu: Whether to use GPU
        """
        self.languages: List[str] = languages
        self.use_gpu: bool = use_gpu
        self.reader: Optional[easyocr.Reader] = None
    
    def _initialize_reader(self) -> None:
        """Initialize EasyOCR reader if not already initialized"""
        if self.reader is None:
            print(f"Initializing EasyOCR ({'+'.join(self.languages)})...")
            self.reader = easyocr.Reader(self.languages, gpu=self.use_gpu, verbose=False)
    
    def _crop_block(self, image: np.ndarray, x: int, y: int, block_w: int, block_h: int, 
                    padding: int) -> np.ndarray:
        """
        Crop a block from image with padding
        
        Returns:
            Tuple of (cropped_image, x1, y1, x2, y2)
        """
        h, w = image.shape[:2]
        
        # Add padding with bounds checking
        x1: int = max(0, x - padding)
        y1: int = max(0, y - padding)
        x2: int = min(w, x + block_w + padding)
        y2: int = min(h, y + block_h + padding)
        
        # Crop block region
        return image[y1:y2, x1:x2]
    
    def recognize_blocks(self, image: np.ndarray, blocks: List[Tuple[int, int, int, int]], 
                         padding: int = 10) -> List[Dict[str, Any]]:
        """
        Recognize text in detected blocks
        
        Args:
            image: Input image (BGR)
            blocks: List of (x, y, w, h) block coordinates
            padding: Padding around each block
            
        Returns:
            List of OCR results
        """
        self._initialize_reader()
        
        if self.reader is None:
            raise RuntimeError("Failed to initialize EasyOCR reader")
        
        h, w = image.shape[:2]
        results: List[Dict[str, Any]] = []
        
        print(f"Recognizing text in {len(blocks)} blocks...")
        print("-" * 50)
        
        for i, (x, y, block_w, block_h) in enumerate(blocks):
            # Crop block with padding
            cropped = self._crop_block(image, x, y, block_w, block_h, padding)
            
            if cropped.size == 0:
                results.append({
                    'box_index': i + 1,
                    'bbox': {'x': x, 'y': y, 'width': block_w, 'height': block_h},
                    'text': '',
                    'confidence': 0.0,
                    'status': 'empty'
                })
                print(f"  Box {i+1}: empty region")
                continue
            
            # Recognize text
            ocr_result: List[Tuple[List[Tuple[int, int]], str, float]] = self.reader.readtext(cropped)
            
            if ocr_result:
                # Combine all found texts
                full_text: str = ' '.join([res[1] for res in ocr_result])
                avg_confidence: float = sum([res[2] for res in ocr_result]) / len(ocr_result)
                
                results.append({
                    'box_index': i + 1,
                    'bbox': {'x': x, 'y': y, 'width': block_w, 'height': block_h},
                    'text': full_text,
                    'confidence': float(avg_confidence),
                    'status': 'recognized',
                    'fragments': len(ocr_result)
                })
                
                print(f"  Box {i+1}: '{full_text}' (conf={avg_confidence:.3f}, fragments={len(ocr_result)})")
            else:
                results.append({
                    'box_index': i + 1,
                    'bbox': {'x': x, 'y': y, 'width': block_w, 'height': block_h},
                    'text': '',
                    'confidence': 0.0,
                    'status': 'no_text'
                })
                print(f"  Box {i+1}: text not found")
        
        print("-" * 50)
        
        # Statistics
        recognized: int = sum(1 for r in results if r['status'] == 'recognized')
        no_text: int = sum(1 for r in results if r['status'] == 'no_text')
        empty: int = sum(1 for r in results if r['status'] == 'empty')
        
        print("OCR STATISTICS:")
        print(f"  - Total blocks: {len(results)}")
        print(f"  - Recognized text: {recognized}")
        print(f"  - Text not found: {no_text}")
        print(f"  - Empty regions: {empty}")
        
        if recognized > 0:
            avg_conf: float = sum(r['confidence'] for r in results if r['status'] == 'recognized') / recognized
            print(f"  - Average confidence: {avg_conf:.3f}")
        
        return results
    
    def save_results(self, results: List[Dict[str, Any]], output_path: Union[str, Path]) -> None:
        """Save OCR results to JSON"""
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2, ensure_ascii=False, default=self._convert_numpy_types)
        print(f"OCR results saved to: {output_path}")
    
    @staticmethod
    def _convert_numpy_types(obj: Any) -> Any:
        """
        Convert numpy types to Python types for JSON serialization
        
        Args:
            obj: Object to convert
            
        Returns:
            JSON-serializable object
            
        Raises:
            TypeError: If object type is not JSON serializable
        """
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, np.bool_):
            return bool(obj)
        raise TypeError(f"Object of type {type(obj)} is not JSON serializable")