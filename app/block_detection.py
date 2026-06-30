"""
Block detection and extraction from images
"""
import cv2
import numpy as np
import random
from typing import List, Tuple, Dict, Any
from scipy.ndimage import morphology as scipy_morph, label


class BlockDetector:
    """Detects rectangular blocks in document images"""
    
    def __init__(self, dilation_size: int = 7, min_size: int = 500):
        """
        Initialize block detector
        
        Args:
            dilation_size: Size of dilation kernel
            min_size: Minimum component size in pixels
        """
        self.dilation_size = dilation_size
        self.min_size = min_size
        self.debug_images = []
        self.debug_names = []
        self.debug_paths = []
    
    def detect_blocks(self, img: np.ndarray, std_multiplier: float) -> Tuple[List[Tuple[int, int, int, int]], List[np.ndarray]]:
        """
        Detect rectangular blocks in image
        
        Args:
            img: Input image (BGR)
            std_multiplier: Multiplier for standard deviation threshold
            
        Returns:
            Tuple of (boxes_coords, debug_images)
        """
        self.debug_images = []
        self.debug_names = []
        self.debug_paths = []
        
        # Convert to grayscale
        if len(img.shape) == 3:
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        else:
            gray = img.copy()
        
        # Keep color version for drawing
        if len(img.shape) == 3:
            img_color = img.copy()
        else:
            img_color = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
        
        # 1. Dilation
        footprint = np.ones((self.dilation_size, self.dilation_size), dtype=bool)
        dilated = scipy_morph.grey_dilation(gray, footprint=footprint)
        self._add_debug(dilated, f"Dilation (kernel={self.dilation_size}x{self.dilation_size})", 
                        f"02_dilation_kernel_{self.dilation_size}")
        
        # 2. Morphological gradient
        gradient = dilated.astype(np.int32) - gray.astype(np.int32)
        gradient_display = np.clip(gradient, 0, 255).astype(np.uint8)
        self._add_debug(gradient_display, "Morphological gradient", "03_morphological_gradient")
        
        # 3. Calculate threshold
        mean_val = gradient.mean()
        std_val = gradient.std()
        threshold = mean_val + (std_val * std_multiplier)
        
        print(f"Gradient statistics:")
        print(f"  - Mean value: {mean_val:.2f}")
        print(f"  - Standard deviation: {std_val:.2f}")
        print(f"  - Threshold (mean + {std_multiplier}*std): {threshold:.2f}")
        
        # 4. Binarization
        binary = gradient.copy()
        binary[binary < threshold] = 0
        binary[binary >= threshold] = 1
        binary_display = (binary * 255).astype(np.uint8)
        self._add_debug(binary_display, f"Binarization (threshold={threshold:.2f})", 
                       f"04_binary_threshold_{threshold:.2f}")
        
        # # 5. Inverted binary
        # binary_inv = cv2.bitwise_not(binary_display)
        # self._add_debug(binary_inv, "Inverted binary", "05_binary_inverted")
        
        # # 6. Morphological operations
        # kernel = np.ones((3, 3), np.uint8)
        # binary_closed = cv2.morphologyEx(binary_display, cv2.MORPH_CLOSE, kernel)
        # self._add_debug(binary_closed, "Morphological closing (3x3)", "06_morphological_close")
        
        # 7. Connected components
        # lbl, numcc = label(binary_closed)
        lbl, numcc = label(binary)
        
        # Label visualization
        lbl_colored = np.zeros((*lbl.shape, 3), dtype=np.uint8)
        component_sizes = []
        for i in range(1, numcc + 1):
            color = (random.randint(50, 255), random.randint(50, 255), random.randint(50, 255))
            lbl_colored[lbl == i] = color
            component_sizes.append(np.sum(lbl == i))
        
        self._add_debug(lbl_colored, f"Connected components (total: {numcc})", 
                       f"07_connected_components_{numcc}")
        
        print(f"Connected components:")
        print(f"  - Total found: {numcc}")
        print(f"  - Minimum size: {self.min_size} pixels")
        
        # 8. Filter components
        boxes_coords = []
        boxes_img = img_color.copy()
        components_filtered = np.zeros_like(lbl_colored)
        filtered_count = 0
        rejected_count = 0
        
        for i in range(1, numcc + 1):
            py, px = np.nonzero(lbl == i)
            component_size = len(py)
            
            if component_size < self.min_size:
                rejected_count += 1
                continue
            
            filtered_count += 1
            components_filtered[lbl == i] = lbl_colored[lbl == i]
            
            xmin, xmax, ymin, ymax = px.min(), px.max(), py.min(), py.max()
            x, y, w, h = xmin, ymin, (xmax - xmin), (ymax - ymin)
            
            boxes_coords.append((x, y, w, h))
            
            # Draw rectangle
            color = (random.randint(100, 255), random.randint(100, 255), random.randint(100, 255))
            cv2.rectangle(boxes_img, (x, y), (x + w, y + h), color, 2)
            
            # Add center
            cx = x + w // 2
            cy = y + h // 2
            cv2.circle(boxes_img, (cx, cy), 4, (0, 255, 255), -1)
            
            # Add number
            cv2.putText(boxes_img, str(filtered_count), (x, y-5),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)
            
            print(f"  Component {filtered_count}: size={component_size}, x={x}, y={y}, w={w}, h={h}")
        
        self._add_debug(components_filtered, f"Filtered components ({filtered_count} of {numcc})", 
                       f"08_filtered_components_{filtered_count}")
        self._add_debug(boxes_img, f"Found rectangles ({filtered_count})", 
                       f"09_final_boxes_{filtered_count}")
        
        print(f"Found {filtered_count} blocks, rejected {rejected_count}")
        
        return boxes_coords, self.debug_images, self.debug_paths
    
    def _add_debug(self, img: np.ndarray, name: str, path: str) -> None:
        """Add image to debug collection"""
        self.debug_images.append(img)
        self.debug_names.append(name)
        self.debug_paths.append(path)