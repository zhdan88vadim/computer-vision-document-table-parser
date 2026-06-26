import cv2
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from scipy.ndimage import morphology as scipy_morph, label
import random

def deskew_image(img):
    """
    Выравнивание изображения (поворот по углу наклона)
    """
    if len(img.shape) == 3:
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    else:
        gray = img.copy()
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    
    # Предобработка
    gray = cv2.GaussianBlur(gray, (5, 5), 0)
    
    # Бинаризация
    thresh = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                   cv2.THRESH_BINARY_INV, 11, 2)
    
    # Поиск контуров
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    if not contours:
        print("Контуры не найдены")
        return img
    
    # Находим самый большой контур
    largest_contour = max(contours, key=cv2.contourArea)
    
    # Определяем угол наклона
    rect = cv2.minAreaRect(largest_contour)
    angle = rect[2]
    
    if angle < -45:
        angle = 90 + angle
    
    # Если угол слишком мал, не поворачиваем
    if abs(angle) < 0.5:
        print(f"Изображение уже ровное, угол наклона: {angle:.2f} градусов")
        return img
    
    print(f"Обнаружен угол наклона: {angle:.2f} градусов")
    
    # Поворот изображения
    (h, w) = img.shape[:2]
    center = (w // 2, h // 2)
    M = cv2.getRotationMatrix2D(center, angle, 1.0)
    
    # Вычисляем новые границы
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

def extract_boxes(img, dilation_size=3, std_multiplier=1.0, min_size=200):
    """
    Извлечение прямоугольных областей из изображения
    """
    # Конвертация в оттенки серого
    if len(img.shape) == 3:
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    else:
        gray = img.copy()
        img_color = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    
    if len(img.shape) == 2:
        img_color = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    else:
        img_color = img.copy()
    
    # Морфологический градиент
    footprint = np.ones((dilation_size, dilation_size), dtype=bool)
    dilated = scipy_morph.grey_dilation(gray, footprint=footprint)
    gradient = dilated.astype(np.int32) - gray.astype(np.int32)
    
    # Динамический порог
    mean, std = gradient.mean(), gradient.std()
    threshold = mean + (std * std_multiplier)
    
    # Бинаризация
    binary = gradient.copy()
    binary[binary < threshold] = 0
    binary[binary >= threshold] = 1
    
    # Связные компоненты
    lbl, numcc = label(binary)
    
    # Фильтрация по размеру и извлечение боксов
    boxes_coords = []
    boxes_img = img_color.copy()
    filtered_count = 0
    
    for i in range(1, numcc + 1):
        py, px = np.nonzero(lbl == i)
        
        if len(py) < min_size:
            continue
        
        filtered_count += 1
        xmin, xmax, ymin, ymax = px.min(), px.max(), py.min(), py.max()
        x, y, w, h = xmin, ymin, (xmax - xmin), (ymax - ymin)
        
        boxes_coords.append((x, y, w, h))
        
        color = (random.randint(100, 255), random.randint(100, 255), random.randint(100, 255))
        cv2.rectangle(boxes_img, (x, y), (x + w, y + h), color, 2)
        
        # Добавляем номер
        cv2.putText(boxes_img, str(filtered_count), (x, y-5), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)
    
    return boxes_img, boxes_coords, filtered_count, numcc, threshold

def process_image(image_path, output_dir="./output/", 
                  dilation_size=3, std_multiplier=1.0, min_size=200,
                  show_results=True):
    """
    Основная функция обработки: выравнивание + извлечение блоков
    """
    # Создаем выходную папку
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    
    # Загрузка изображения
    img = cv2.imread(image_path)
    if img is None:
        print(f"Ошибка: не удалось загрузить изображение {image_path}")
        return None, None
    
    print(f"Обработка изображения: {image_path}")
    
    # 1. Выравнивание изображения
    print("Шаг 1: Выравнивание изображения...")
    img_aligned = deskew_image(img)
    
    # Сохраняем выравненное изображение
    aligned_path = f"{output_dir}/aligned_{Path(image_path).name}"
    cv2.imwrite(aligned_path, img_aligned)
    print(f"Выравненное изображение сохранено: {aligned_path}")
    
    # 2. Извлечение блоков
    print("Шаг 2: Извлечение прямоугольных блоков...")
    boxes_img, boxes_coords, filtered_count, total_count, threshold = extract_boxes(
        img_aligned, dilation_size, std_multiplier, min_size
    )
    
    # Сохраняем результат с боксами
    boxes_path = f"{output_dir}/boxes_{Path(image_path).name}"
    cv2.imwrite(boxes_path, boxes_img)
    print(f"Изображение с боксами сохранено: {boxes_path}")
    
    # 3. Статистика
    print("\n" + "="*50)
    print("📊 СТАТИСТИКА:")
    print(f"   - Всего компонент (блоков) найдено: {total_count}")
    print(f"   - Отфильтровано и нарисовано: {filtered_count}")
    print(f"   - Параметры: Dilation={dilation_size}, Std Multiplier={std_multiplier:.2f}, MinSize={min_size}")
    print(f"   - Порог бинаризации: {threshold:.2f}")
    
    if boxes_coords:
        print("\n📝 НАЙДЕННЫЕ ПРЯМОУГОЛЬНИКИ:")
        for i, (x, y, w, h) in enumerate(boxes_coords):
            area = w * h
            print(f"   {i+1}: x={x}, y={y}, w={w}, h={h}, площадь={area}")
    else:
        print("\n⚠️ Прямоугольники не найдены")
    
    print("="*50)
    
    # Отображение результатов
    if show_results:
        fig, axes = plt.subplots(1, 3, figsize=(18, 6))
        
        # Исходное
        axes[0].imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        axes[0].set_title("Исходное изображение")
        axes[0].axis('off')
        
        # Выравненное
        axes[1].imshow(cv2.cvtColor(img_aligned, cv2.COLOR_BGR2RGB))
        axes[1].set_title("После выравнивания")
        axes[1].axis('off')
        
        # С боксами
        axes[2].imshow(cv2.cvtColor(boxes_img, cv2.COLOR_BGR2RGB))
        axes[2].set_title(f"Найдено блоков: {filtered_count}")
        axes[2].axis('off')
        
        plt.tight_layout()
        plt.show()
    
    return boxes_img, boxes_coords

# ==================== ЗАПУСК ====================
if __name__ == "__main__":
    # Параметры обработки
    input_image = "/media/vadim/1TB_SSD/my_github/computer-vision-document-table-parser/input_images/1.jpg"
    
    # Настройки для извлечения блоков
    dilation_size = 3       # Размер ядра дилатации
    std_multiplier = 1.0    # Множитель стандартного отклонения для порога
    min_size = 200          # Минимальный размер блока в пикселях
    
    # Запуск обработки
    result_img, boxes = process_image(
        image_path=input_image,
        output_dir="./output_finish/",
        dilation_size=dilation_size,
        std_multiplier=std_multiplier,
        min_size=min_size,
        show_results=True
    )