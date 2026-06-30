"""
Image alignment and deskewing
"""
import cv2
import numpy as np


class ImageAligner:
    """Handles image deskewing and alignment"""
    
    def deskew_image(self, img: np.ndarray) -> np.ndarray:
        """
        Align image by rotating it based on the skew angle
        
        Args:
            img: Input image (BGR or grayscale)
            
        Returns:
            Deskewed image
        """
        if len(img.shape) == 3:
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        else:
            gray = img.copy()
            img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
        
        # Preprocessing
        gray = cv2.GaussianBlur(gray, (5, 5), 0)
        
        # Binarization
        thresh = cv2.adaptiveThreshold(
            gray, 255, 
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY_INV, 11, 2
        )
        
        # Find contours
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        if not contours:
            print("Contours not found")
            return img
        
        # Find the largest contour
        largest_contour = max(contours, key=cv2.contourArea)
        
        # Determine skew angle
        rect = cv2.minAreaRect(largest_contour)
        angle = rect[2]
        
        if angle < -45:
            angle = 90 + angle
        
        print(f"Detected skew angle: {angle:.2f} degrees")
        
        # Rotate image
        (h, w) = img.shape[:2]
        center = (w // 2, h // 2)
        M = cv2.getRotationMatrix2D(center, angle, 1.0)
        
        # Calculate new boundaries
        cos = np.abs(M[0, 0])
        sin = np.abs(M[0, 1])
        new_w = int((h * sin) + (w * cos))
        new_h = int((h * cos) + (w * sin))
        
        M[0, 2] += (new_w / 2) - center[0]
        M[1, 2] += (new_h / 2) - center[1]
        
        rotated = cv2.warpAffine(
            img, M, (new_w, new_h),
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=(255, 255, 255)
        )
        
        return rotated