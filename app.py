import cv2
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from scipy.ndimage import morphology as scipy_morph, label
import random
import os
import shutil
import json
import easyocr
from PIL import Image, ImageDraw, ImageFont
from fuzzywuzzy import fuzz


def deskew_image(img):
    """
    Align image by rotating it based on the skew angle
    """
    if len(img.shape) == 3:
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    else:
        gray = img.copy()
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)

    # Preprocessing
    gray = cv2.GaussianBlur(gray, (5, 5), 0)

    # Binarization
    thresh = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                   cv2.THRESH_BINARY_INV, 11, 2)

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

    rotated = cv2.warpAffine(img, M, (new_w, new_h),
                            borderMode=cv2.BORDER_CONSTANT,
                            borderValue=(255, 255, 255))

    return rotated


def clean_output_folder(output_dir):
    """
    Clean output folder by removing all contents
    """
    output_path = Path(output_dir)

    if output_path.exists():
        for item in output_path.glob('*'):
            if item.is_file():
                item.unlink()
            elif item.is_dir():
                shutil.rmtree(item)
        print(f"Folder {output_dir} cleaned")
    else:
        output_path.mkdir(parents=True, exist_ok=True)
        print(f"Folder {output_dir} created")


def map_boxes_to_fields(ocr_results, field_mapping):
    """
    Map box numbers to field names

    Args:
        ocr_results: list of OCR results
        field_mapping: dictionary {box_number: field_name}

    Returns:
        dict: dictionary {field_name: text}
    """
    mapped_data = {}

    for result in ocr_results:
        box_index = result['box_index']
        text = result['text']

        # If mapping exists for this box
        if box_index in field_mapping:
            field_name = field_mapping[box_index]
            mapped_data[field_name] = text
        else:
            # If no mapping, use box_index as name
            mapped_data[f"box_{box_index}"] = text

    return mapped_data


def save_mapped_results(img_name, mapped_data, output_dir="./output_finish/"):
    """
    Save mapped results to JSON
    """
    json_path = os.path.join(output_dir, f"{img_name}_mapped_results.json")

    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump(mapped_data, f, indent=2, ensure_ascii=False)
    print(f"Mapped results saved to: {json_path}")

    return mapped_data


def recognize_boxes_with_easyocr(image, boxes_coords, output_dir="./output_finish/", padding=15):
    """
    Recognize text in detected boxes using EasyOCR
    """
    # Initialize EasyOCR
    print("\nInitializing EasyOCR (Russian + English)...")
    reader = easyocr.Reader(['ru', 'en'], gpu=False, verbose=False)

    h, w = image.shape[:2]
    results = []

    print(f"\nRecognizing text in {len(boxes_coords)} boxes...")
    print("-" * 50)

    for i, (x, y, box_w, box_h) in enumerate(boxes_coords):
        # Add padding
        x1 = max(0, x - padding)
        y1 = max(0, y - padding)
        x2 = min(w, x + box_w + padding)
        y2 = min(h, y + box_h + padding)

        # Crop box region
        cropped = image[y1:y2, x1:x2]

        if cropped.size == 0:
            results.append({
                'box_index': i + 1,
                'bbox': {'x': x, 'y': y, 'width': box_w, 'height': box_h},
                'text': '',
                'confidence': 0.0,
                'status': 'empty'
            })
            print(f"   Box {i+1}: empty region")
            continue

        # Recognize text
        ocr_result = reader.readtext(cropped)

        if ocr_result:
            # Combine all found texts
            full_text = ' '.join([res[1] for res in ocr_result])
            avg_confidence = sum([res[2] for res in ocr_result]) / len(ocr_result)

            results.append({
                'box_index': i + 1,
                'bbox': {'x': x, 'y': y, 'width': box_w, 'height': box_h},
                'text': full_text,
                'confidence': float(avg_confidence),
                'status': 'recognized',
                'fragments': len(ocr_result)
            })

            print(f"   Box {i+1}: '{full_text}' (conf={avg_confidence:.3f}, fragments={len(ocr_result)})")
        else:
            results.append({
                'box_index': i + 1,
                'bbox': {'x': x, 'y': y, 'width': box_w, 'height': box_h},
                'text': '',
                'confidence': 0.0,
                'status': 'no_text'
            })
            print(f"   Box {i+1}: text not found")

    print("-" * 50)

    # Save results to JSON
    json_path = os.path.join(output_dir, "ocr_results.json")
    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False, default=convert_numpy_types)
    print(f"Results saved to: {json_path}")

    return results


def convert_numpy_types(obj):
    """Convert numpy types to Python types for JSON serialization"""
    if isinstance(obj, np.integer):
        return int(obj)
    elif isinstance(obj, np.floating):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    raise TypeError(f"Object of type {type(obj)} is not JSON serializable")


def visualize_ocr_results(img_name, image, boxes_coords, ocr_results, output_dir="./output_finish/"):
    """
    Visualize OCR results on the image
    """
    pil_img = Image.fromarray(image)
    draw = ImageDraw.Draw(pil_img)

    # Load font
    try:
        font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 16)
    except:
        font = ImageFont.load_default()
        print("Warning: Cyrillic font not found, using default")

    # Draw boxes and text
    for i, (box, result) in enumerate(zip(boxes_coords, ocr_results)):
        x, y, w, h = box
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
        # Get text size
        bbox = draw.textbbox((0, 0), label, font=font)
        text_width = bbox[2] - bbox[0]
        text_height = bbox[3] - bbox[1]

        # Draw background for text
        draw.rectangle([(x1, y1 - text_height), (x1 + text_width + 2, y1+10)],
                    fill=(200, 200, 200))

        # Draw text
        draw.text((x1, y1-10), label, fill=(0, 0, 0), font=font)

    # Save result
    img_result = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)
    output_path = os.path.join(output_dir, f"{img_name}_ocr_visualization.jpg")
    cv2.imwrite(output_path, img_result)
    print(f"Visualization saved: {output_path}")


def load_expected_data_for_image(image_name, expected_dir="./test_data/expected"):
    json_path = Path(expected_dir) / f"{image_name}.json"
    if not json_path.exists():
        raise FileNotFoundError(f"Expected results file not found: {json_path}")

    with open(json_path, 'r', encoding='utf-8') as f:
        return json.load(f)


def compare_results_fuzzy(actual, expected, min_score=80):
    """Compare results using fuzzywuzzy"""
    errors = []
    details = {}

    for key in expected:
        if key not in actual:
            errors.append(f"Missing field: {key}")
            details[key] = {'status': 'missing'}
            continue

        exp = expected[key] or ""
        act = actual[key] or ""

        # If both are empty - ok
        if not exp and not act:
            details[key] = {'status': 'match', 'score': 100}
            continue

        # Calculate similarity using different methods
        score_ratio = fuzz.ratio(act, exp)
        score_partial = fuzz.partial_ratio(act, exp)
        score_token = fuzz.token_sort_ratio(act, exp)

        # Take maximum
        score = max(score_ratio, score_partial, score_token)

        details[key] = {
            'status': 'match' if score >= min_score else 'mismatch',
            'score': score,
            'expected': exp,
            'actual': act
        }

        if score < min_score:
            errors.append(
                f"{key}: expected '{exp}', got '{act}' (similarity: {score}%)"
            )

    return len(errors) == 0, errors, details


def process_image_with_ocr(img_path, dilation_size=3, std_multiplier=1.0, min_size=200,
                           output_dir="./output_finish/", show_plots=True, save_images=True,
                           ocr_padding=10):
    """
    Full image processing: alignment, block extraction, and OCR
    """
    # Create output folders
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    debug_dir = os.path.join(output_dir, "debug")
    Path(debug_dir).mkdir(parents=True, exist_ok=True)

    # 1. Load image
    img = cv2.imread(img_path)
    if img is None:
        print(f"Error: Could not load image {img_path}")
        return None, None

    img_name = Path(img_path).stem
    print(f"Processing image: {img_path}")
    print("="*60)

    # ============ STEP 1: ALIGNMENT ============
    print("\nSTEP 1: IMAGE ALIGNMENT")
    print("-" * 40)

    img_aligned = deskew_image(img)

    # Save aligned image
    aligned_path = os.path.join(output_dir, f"{img_name}_aligned.jpg")
    cv2.imwrite(aligned_path, img_aligned)
    print(f"Aligned image saved: {aligned_path}")

    # ============ STEP 2: BLOCK EXTRACTION ============
    print("\nSTEP 2: RECTANGULAR BLOCK EXTRACTION")
    print("-" * 40)

    # Convert to grayscale
    if len(img_aligned.shape) == 3:
        gray = cv2.cvtColor(img_aligned, cv2.COLOR_BGR2GRAY)
    else:
        gray = img_aligned.copy()

    # Keep color version for drawing
    if len(img_aligned.shape) == 3:
        img_color = img_aligned.copy()
    else:
        img_color = cv2.cvtColor(img_aligned, cv2.COLOR_GRAY2BGR)

    # ============ SAVE ALL INTERMEDIATE STEPS ============
    debug_images = []
    debug_names = []
    debug_paths = []

    # 1. Original image (grayscale)
    debug_images.append(gray.copy())
    debug_names.append("1. Original image (grayscale)")
    debug_paths.append("01_original_gray")

    # 2. Dilation
    footprint = np.ones((dilation_size, dilation_size), dtype=bool)
    dilated = scipy_morph.grey_dilation(gray, footprint=footprint)
    debug_images.append(dilated.copy())
    debug_names.append(f"2. Dilation (kernel={dilation_size}x{dilation_size})")
    debug_paths.append(f"02_dilation_kernel_{dilation_size}")

    # 3. Morphological gradient
    gradient = dilated.astype(np.int32) - gray.astype(np.int32)
    gradient_display = np.clip(gradient, 0, 255).astype(np.uint8)
    debug_images.append(gradient_display)
    debug_names.append("3. Morphological gradient")
    debug_paths.append("03_morphological_gradient")

    # 4. Gradient statistics
    mean_val = gradient.mean()
    std_val = gradient.std()
    threshold = mean_val + (std_val * std_multiplier)

    print(f"   Gradient statistics:")
    print(f"   - Mean value: {mean_val:.2f}")
    print(f"   - Standard deviation: {std_val:.2f}")
    print(f"   - Threshold (mean + {std_multiplier}*std): {threshold:.2f}")

    # 5. Binarization
    binary = gradient.copy()
    binary[binary < threshold] = 0
    binary[binary >= threshold] = 1
    binary_display = (binary * 255).astype(np.uint8)
    debug_images.append(binary_display)
    debug_names.append(f"4. Binarization (threshold={threshold:.2f})")
    debug_paths.append(f"04_binary_threshold_{threshold:.2f}")

    # 6. Inverted binary
    binary_inv = cv2.bitwise_not(binary_display)
    debug_images.append(binary_inv)
    debug_names.append("5. Inverted binary")
    debug_paths.append("05_binary_inverted")

    # 7. Morphological operations for improvement
    kernel = np.ones((3, 3), np.uint8)
    binary_closed = cv2.morphologyEx(binary_display, cv2.MORPH_CLOSE, kernel)
    debug_images.append(binary_closed)
    debug_names.append("6. Morphological closing (3x3)")
    debug_paths.append("06_morphological_close")

    # 8. Connected components
    lbl, numcc = label(binary)

    # Label visualization
    lbl_colored = np.zeros((*lbl.shape, 3), dtype=np.uint8)
    component_sizes = []
    for i in range(1, numcc + 1):
        color = (random.randint(50, 255), random.randint(50, 255), random.randint(50, 255))
        lbl_colored[lbl == i] = color
        component_sizes.append(np.sum(lbl == i))

    debug_images.append(lbl_colored)
    debug_names.append(f"7. Connected components (total: {numcc})")
    debug_paths.append(f"07_connected_components_{numcc}")

    print(f"\n   Connected components:")
    print(f"   - Total found: {numcc}")
    print(f"   - Minimum size: {min_size} pixels")

    # 9. Filter components
    boxes_coords = []
    boxes_img = img_color.copy()
    components_filtered = np.zeros_like(lbl_colored)
    filtered_count = 0
    rejected_count = 0

    for i in range(1, numcc + 1):
        py, px = np.nonzero(lbl == i)
        component_size = len(py)

        if component_size < min_size:
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

        print(f"   Component {filtered_count}: size={component_size}, x={x}, y={y}, w={w}, h={h}")

    # Save visualizations
    debug_images.append(components_filtered)
    debug_names.append(f"8. Filtered components ({filtered_count} of {numcc})")
    debug_paths.append(f"08_filtered_components_{filtered_count}")

    debug_images.append(boxes_img)
    debug_names.append(f"9. Found rectangles ({filtered_count})")
    debug_paths.append(f"09_final_boxes_{filtered_count}")

    # Save debug images
    if save_images:
        print(f"\nSaving debug images to: {debug_dir}")
        for debug_img, debug_path, debug_name in zip(debug_images, debug_paths, debug_names):
            save_path = os.path.join(debug_dir, f"{img_name}_{debug_path}.jpg")
            cv2.imwrite(save_path, debug_img)
        print(f"Saved {len(debug_images)} debug images")

    # Save result with boxes
    boxes_path = os.path.join(output_dir, f"{img_name}_boxes.jpg")
    cv2.imwrite(boxes_path, boxes_img)
    print(f"Result with boxes saved: {boxes_path}")

    # ============ STEP 3: TEXT RECOGNITION ============
    print("\nSTEP 3: TEXT RECOGNITION (EasyOCR)")
    print("-" * 40)

    if boxes_coords:
        # Recognize text
        ocr_results = recognize_boxes_with_easyocr(
            img_aligned,  # Use aligned image
            boxes_coords,
            output_dir=output_dir,
            padding=ocr_padding
        )

        # Visualize OCR results
        visualize_ocr_results(
            img_name,
            img_aligned,
            boxes_coords,
            ocr_results,
            output_dir=output_dir
        )

        # Extra fields
        extra_fields = {'Рассмотрение на рабочей комиссии': ''}

        # Define mapping (configure manually)
        field_mapping = {
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
        }

        # Apply mapping
        mapped_data = map_boxes_to_fields(ocr_results, field_mapping)
        mapped_data.update(extra_fields)

        # Save mapped results
        save_mapped_results(img_name, mapped_data, output_dir)

        quick_test(img_name, mapped_data)

        # Output OCR statistics
        recognized = sum(1 for r in ocr_results if r['status'] == 'recognized')
        no_text = sum(1 for r in ocr_results if r['status'] == 'no_text')
        empty = sum(1 for r in ocr_results if r['status'] == 'empty')

        print(f"\nOCR STATISTICS:")
        print(f"   - Total boxes: {len(ocr_results)}")
        print(f"   - Recognized text: {recognized}")
        print(f"   - Text not found: {no_text}")
        print(f"   - Empty regions: {empty}")

        if recognized > 0:
            avg_conf = sum(r['confidence'] for r in ocr_results if r['status'] == 'recognized') / recognized
            print(f"   - Average confidence: {avg_conf:.3f}")
    else:
        print("Warning: Boxes not found, OCR skipped")
        ocr_results = []

    # ============ STATISTICS ============
    print("\n" + "="*60)
    print("GENERAL STATISTICS:")
    print(f"   - Found blocks: {filtered_count}")
    print(f"   - Rejected blocks: {rejected_count}")
    print(f"   - Parameters: Dilation={dilation_size}, Std Multiplier={std_multiplier:.2f}, MinSize={min_size}")
    if boxes_coords and ocr_results:
        recognized = sum(1 for r in ocr_results if r['status'] == 'recognized')
        print(f"   - Recognized text: {recognized}/{len(ocr_results)}")
    print("="*60)

    # ============ VISUALIZATION ============
    if show_plots:
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

        if save_images:
            summary_path = os.path.join(output_dir, f"{img_name}_debug_summary.png")
            fig.savefig(summary_path, dpi=150, bbox_inches='tight')
            print(f"Debug summary saved: {summary_path}")

        plt.show()

    return boxes_coords, ocr_results


def quick_test(img_name, actual_data, expected_dir="./test_data/expected/", min_score=80):
    """Quick test with saving and output"""

    try:
        expected_data = load_expected_data_for_image(img_name, expected_dir)
        success, errors, details = compare_results_fuzzy(actual_data, expected_data, min_score)
        
        print(f"\n{'✅' if success else '❌'} Тест {img_name}: {'ПРОЙДЕН' if success else 'НЕ ПРОЙДЕН'}")
        
        if errors:
            print("   ERRORS:")
            for error in errors:
                print(f"   - {error}")

        # Statistics
        total = len(details)
        matches = sum(1 for d in details.values() if d['status'] == 'match')
        mismatches = sum(1 for d in details.values() if d['status'] == 'mismatch')

        print(f"\n   STATISTICS:")
        print(f"   - Total fields: {total}")
        print(f"   - Matches: {matches}")
        print(f"   - Mismatches: {mismatches}")

        if matches > 0:
            avg_score = sum(d['score'] for d in details.values() if d['status'] == 'match') / matches
            print(f"   - Average similarity: {avg_score:.1f}%")

        # Save result
        result = {
            "image": img_name,
            "success": success,
            "min_score": min_score,
            "errors": errors,
            "details": details,
            "stats": {
                "total": total,
                "matches": matches,
                "mismatches": mismatches
            }
        }

        output_path = f"./test_data/output/{img_name}_result.json"
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(result, f, indent=2, ensure_ascii=False)

        print(f"\nResult saved: {output_path}")

        return success

    except Exception as e:
        print(f"❌ Ошибка: {e}")
        return False


def batch_process(input_dir, output_dir, ocr_padding,
                  dilation_size=3, std_multiplier=1.0, min_size=500,
                  show_plots=False, save_images=True):
    """
    Process all images in a folder
    """
    input_path = Path(input_dir)
    if not input_path.exists():
        print(f"Error: folder {input_dir} does not exist")
        return

    # Supported formats
    extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff'}
    image_files = [f for f in input_path.iterdir() if f.suffix.lower() in extensions]

    if not image_files:
        print(f"No images found in {input_dir}")
        return

    print(f"Found {len(image_files)} images to process")
    print("="*60)

    results = {}
    for img_file in image_files:
        print(f"\nProcessing: {img_file.name}")
        boxes_img, boxes_coords = process_image_with_ocr(
            img_path=img_file,
            dilation_size=dilation_size,
            std_multiplier=std_multiplier,
            min_size=min_size,
            output_dir=output_dir,
            show_plots=show_plots,
            save_images=save_images,
            ocr_padding=ocr_padding
        )
        results[img_file.name] = {
            'boxes_img': boxes_img,
            'boxes_coords': boxes_coords,
            'num_boxes': len(boxes_coords) if boxes_coords else 0
        }

    # Overall statistics
    print("\n" + "="*60)
    print("PROCESSING SUMMARY:")
    for name, data in results.items():
        print(f"   {name}: found {data['num_boxes']} blocks")
    print("="*60)

    return results


# ==================== MAIN ============
if __name__ == "__main__":
    # Processing parameters
    input_dir = "./input_images"

    # Settings for block extraction
    dilation_size = 7
    std_multiplier = 1.0
    min_size = 500
    ocr_padding = 10  # Padding for OCR

    # Clean output folder
    clean_output_folder("./output_finish/")

    batch_process(input_dir=input_dir,
                  output_dir="./output_finish/",
                  dilation_size=dilation_size,
                  min_size=min_size,
                  ocr_padding=ocr_padding
                  )