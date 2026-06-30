"""
Visualization utilities
"""
import cv2
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from PIL import Image, ImageDraw, ImageFont
from typing import List, Dict, Any, Tuple, Optional


class Visualizer:
    """Handles visualization of processing results"""
    
    def __init__(self):
        self.font_path = "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"
    
    def draw_boxes(self, image: np.ndarray, blocks: List[Tuple[int, int, int, int]], 
                   ocr_results: List[Dict[str, Any]], output_path: Path) -> None:
        """
        Draw boxes and OCR results on image
        
        Args:
            image: Input image (BGR)
            blocks: List of (x, y, w, h) block coordinates
            ocr_results: List of OCR results
            output_path: Path to save visualization
        """
        pil_img = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        draw = ImageDraw.Draw(pil_img)
        
        # Load font
        try:
            font = ImageFont.truetype(self.font_path, 16)
        except:
            font = ImageFont.load_default()
            print("Warning: Cyrillic font not found, using default")
        
        for i, (block, result) in enumerate(zip(blocks, ocr_results)):
            x, y, w, h = block
            x1, y1, x2, y2 = x, y, x + w, y + h
            
            # Draw rectangle
            color = (0, 255, 0) if result['text'] else (255, 0, 0)
            draw.rectangle([(x1, y1), (x2, y2)], outline=color, width=2)
            
            # Format text for display
            if result['text']:
                label = f"#{i+1}: {result['text']} ({result['confidence']:.2f})"
            else:
                label = f"#{i+1}: text not found"
            
            # Place text on top
            bbox = draw.textbbox((0, 0), label, font=font)
            text_width = bbox[2] - bbox[0]
            text_height = bbox[3] - bbox[1]
            
            # Draw background for text
            draw.rectangle([(x1, y1 - text_height), (x1 + text_width + 2, y1 + 10)],
                          fill=(200, 200, 200))
            
            # Draw text
            draw.text((x1, y1 - 10), label, fill=(0, 0, 0), font=font)
        
        # Save result
        img_result = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)
        cv2.imwrite(str(output_path), img_result)
        print(f"Visualization saved: {output_path}")
    
    def save_image(self, image: np.ndarray, output_path: Path) -> None:
        """Save image to file"""
        cv2.imwrite(str(output_path), image)
        print(f"Image saved: {output_path}")
    
    def save_debug_images(self, debug_images: List[np.ndarray], debug_dir: Path, img_name: str) -> None:
        """Save debug images to directory"""
        debug_names = []
        
        # Save individual debug images
        for i, img in enumerate(debug_images):
            save_path = debug_dir / f"{img_name}_debug_{i:02d}.jpg"
            cv2.imwrite(str(save_path), img)
            debug_names.append(save_path.name)
        
        print(f"Saved {len(debug_images)} debug images to: {debug_dir}")
    
    def create_summary_plot(self, debug_images: List[np.ndarray], debug_names: List[str], 
                           output_path: Path) -> None:
        """
        Create summary plot of debug images
        
        Args:
            debug_images: List of debug images
            debug_names: List of debug image names
            output_path: Path to save plot
        """
        n_images = len(debug_images)
        cols = min(3, n_images)
        rows = (n_images + cols - 1) // cols
        
        fig, axes = plt.subplots(rows, cols, figsize=(6*cols, 5*rows))
        if n_images == 1:
            axes = np.array([axes])
        axes = axes.flatten()
        
        for i, (debug_img, debug_name) in enumerate(zip(debug_images, debug_names)):
            if i < len(axes):
                if len(debug_img.shape) == 3:
                    axes[i].imshow(cv2.cvtColor(debug_img, cv2.COLOR_BGR2RGB))
                else:
                    axes[i].imshow(debug_img, cmap='gray')
                
                axes[i].set_title(debug_name, fontsize=12, fontweight='bold')
                axes[i].axis('off')
        
        for i in range(n_images, len(axes)):
            axes[i].axis('off')
        
        plt.tight_layout()
        fig.savefig(str(output_path), dpi=150, bbox_inches='tight')
        plt.close(fig)
        print(f"Debug summary saved: {output_path}")