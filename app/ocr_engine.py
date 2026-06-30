"""
OCR processing with EasyOCR
"""
import numpy as np
import json
from typing import List, Dict, Any, Tuple
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
        self.languages = languages
        self.use_gpu = use_gpu
        self.reader = None
    
    def _initialize_reader(self):
        """Initialize EasyOCR reader if not already initialized"""
        if self.reader is None:
            print(f"Initializing EasyOCR ({'+'.join(self.languages)})...")
            self.reader = easyocr.Reader(self.languages, gpu=self.use_gpu, verbose=False)
    
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
        
        h, w = image.shape[:2]
        results = []
        
        print(f"Recognizing text in {len(blocks)} blocks...")
        print("-" * 50)
        
        for i, (x, y, block_w, block_h) in enumerate(blocks):
            # Add padding
            x1 = max(0, x - padding)
            y1 = max(0, y - padding)
            x2 = min(w, x + block_w + padding)
            y2 = min(h, y + block_h + padding)
            
            # Crop block region
            cropped = image[y1:y2, x1:x2]
            
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
            ocr_result = self.reader.readtext(cropped)
            
            if ocr_result:
                # Combine all found texts
                full_text = ' '.join([res[1] for res in ocr_result])
                avg_confidence = sum([res[2] for res in ocr_result]) / len(ocr_result)
                
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
        recognized = sum(1 for r in results if r['status'] == 'recognized')
        no_text = sum(1 for r in results if r['status'] == 'no_text')
        empty = sum(1 for r in results if r['status'] == 'empty')
        
        print(f"OCR STATISTICS:")
        print(f"  - Total blocks: {len(results)}")
        print(f"  - Recognized text: {recognized}")
        print(f"  - Text not found: {no_text}")
        print(f"  - Empty regions: {empty}")
        
        if recognized > 0:
            avg_conf = sum(r['confidence'] for r in results if r['status'] == 'recognized') / recognized
            print(f"  - Average confidence: {avg_conf:.3f}")
        
        return results
    
    def save_results(self, results: List[Dict[str, Any]], output_path: str) -> None:
        """Save OCR results to JSON"""
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2, ensure_ascii=False, default=self._convert_numpy_types)
        print(f"OCR results saved to: {output_path}")
    
    @staticmethod
    def _convert_numpy_types(obj):
        """Convert numpy types to Python types for JSON serialization"""
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        raise TypeError(f"Object of type {type(obj)} is not JSON serializable")