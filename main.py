"""
Main document parser application
"""
import cv2
from pathlib import Path
from typing import Dict, Any

from app.config import Config
from app.alignment import ImageAligner
from app.block_detection import BlockDetector
from app.ocr_engine import OCRProcessor
from app.field_mapper import FieldMapper
from app.visualization import Visualizer
from app.testing import Tester
from app.utils import clean_output_folder, get_image_files


class DocumentParser:
    """Main orchestrator for document parsing"""
    
    def __init__(self, config: Config):
        """
        Initialize document parser
        
        Args:
            config: Configuration object
        """
        self.config = config
        self.aligner = ImageAligner()
        self.detector = BlockDetector(config.dilation_size, config.min_size)
        self.ocr = OCRProcessor(config.ocr_languages, config.use_gpu)
        self.mapper = FieldMapper(config.field_mapping, config.extra_fields)
        self.visualizer = Visualizer()
        self.tester = Tester(config.expected_dir, config.test_output_dir, config.min_similarity_score)
        
        # Create directories
        self.config.output_dir.mkdir(parents=True, exist_ok=True)
        self.config.debug_dir.mkdir(parents=True, exist_ok=True)
    
    def process_image(self, img_path: Path) -> Dict[str, Any]:
        """
        Process a single image through the entire pipeline
        
        Args:
            img_path: Path to image file
            
        Returns:
            Dictionary with processing results
        """
        # Load image
        img = cv2.imread(str(img_path))
        if img is None:
            raise ValueError(f"Could not load image: {img_path}")
        
        img_name = img_path.stem
        print(f"\n{'='*60}")
        print(f"Processing: {img_name}")
        print("="*60)
        
        # Step 1: Align
        print("\nSTEP 1: IMAGE ALIGNMENT")
        print("-" * 40)
        img_aligned = self.aligner.deskew_image(img)
        self.visualizer.save_image(img_aligned, self.config.debug_dir / f"{img_name}_aligned.jpg")
        
        # Step 2: Detect blocks
        print("\nSTEP 2: RECTANGULAR BLOCK EXTRACTION")
        print("-" * 40)
        blocks, debug_images, debug_paths = self.detector.detect_blocks(img_aligned, self.config.std_multiplier)
        self.visualizer.save_debug_images(debug_images, debug_paths, self.config.debug_dir, img_name)
        
        # Create summary plot
        if hasattr(self.detector, 'debug_names'):
            self.visualizer.create_summary_plot(
                debug_images, 
                self.detector.debug_names,
                self.config.debug_dir / f"{img_name}_summary.png"
            )
        
        # Step 3: OCR
        print("\nSTEP 3: TEXT RECOGNITION (EasyOCR)")
        print("-" * 40)
        ocr_results = self.ocr.recognize_blocks(img_aligned, blocks, self.config.ocr_padding)
        self.ocr.save_results(ocr_results, str(self.config.output_dir / "ocr_results.json"))
        
        # Step 4: Map fields
        print("\nSTEP 4: FIELD MAPPING")
        print("-" * 40)
        mapped_data = self.mapper.map_fields(ocr_results)
        self.mapper.save_mapped_results(img_name, mapped_data, self.config.output_dir)
        
        # Step 5: Visualize
        print("\nSTEP 5: VISUALIZATION")
        print("-" * 40)
        self.visualizer.draw_boxes(
            img_aligned, 
            blocks, 
            ocr_results,
            self.config.output_dir / f"{img_name}_ocr_visualization.jpg"
        )
        
        # Step 6: Test (if expected data exists)
        print("\nSTEP 6: TESTING")
        print("-" * 40)
        test_result = self.tester.run_tests(img_name, mapped_data)
        
        return {
            'image_name': img_name,
            'blocks': blocks,
            'ocr_results': ocr_results,
            'mapped_data': mapped_data,
            'test_result': test_result
        }
    
    def process_batch(self, input_dir: Path) -> Dict[str, Any]:
        """
        Process all images in a directory
        
        Args:
            input_dir: Directory containing images
            
        Returns:
            Dictionary with processing results for all images
        """
        image_files = get_image_files(input_dir)
        
        if not image_files:
            print(f"No images found in {input_dir}")
            return {}
        
        print(f"Found {len(image_files)} images to process")
        
        results = {}
        for img_path in image_files:
            try:
                results[img_path.name] = self.process_image(img_path)
            except Exception as e:
                print(f"Error processing {img_path.name}: {e}")
                results[img_path.name] = {'error': str(e)}
        
        return results


def main() -> None:
    """Main entry point"""
    # Load configuration
    config = Config()
    
    # Clean output directory
    clean_output_folder(config.output_dir)
    
    # Initialize parser
    parser = DocumentParser(config)
    
    # Process images
    results = parser.process_batch(config.input_dir)
    
    # Print summary
    print("\n" + "="*60)
    print("PROCESSING COMPLETE")
    print(f"Processed {len(results)} images")
    print("="*60)
    
    # Summary statistics
    successful = sum(1 for r in results.values() if 'error' not in r)
    failed = len(results) - successful
    print(f"Successful: {successful}")
    print(f"Failed: {failed}")
    
    # Test summary
    passed = sum(1 for r in results.values() 
                if 'test_result' in r and r['test_result'].get('success', False))
    print(f"Tests passed: {passed}")
    print("="*60)


if __name__ == "__main__":
    main()