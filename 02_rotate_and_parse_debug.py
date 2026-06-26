import cv2
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from scipy.ndimage import morphology as scipy_morph, label
import random
import os
import shutil

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

def clean_output_folder(output_dir):
    """
    Очистка выходной папки (удаление всего содержимого)
    """
    output_path = Path(output_dir)
    
    if output_path.exists():
        # Удаляем все содержимое, но оставляем саму папку
        for item in output_path.glob('*'):
            if item.is_file():
                item.unlink()
            elif item.is_dir():
                shutil.rmtree(item)
        print(f"✓ Папка {output_dir} очищена")
    else:
        output_path.mkdir(parents=True, exist_ok=True)
        print(f"✓ Папка {output_dir} создана")

def process_image_with_debug(img_path, dilation_size=3, std_multiplier=1.0, min_size=200, 
                            output_dir="./output/", show_plots=True, save_images=True):
    """
    Полная обработка изображения с выравниванием и извлечением блоков
    С расширенным дебаггингом всех этапов
    """
    # Создаем выходные папки
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    debug_dir = os.path.join(output_dir, "debug")
    Path(debug_dir).mkdir(parents=True, exist_ok=True)
    
    # 1. Загрузка изображения
    img = cv2.imread(img_path)
    if img is None:
        print(f"Ошибка: не удалось загрузить изображение {img_path}")
        return None, None
    
    img_name = Path(img_path).stem
    print(f"Обработка изображения: {img_path}")
    print("="*60)
    
    # ============ ЭТАП 1: ВЫРАВНИВАНИЕ ============
    print("\n📐 ЭТАП 1: ВЫРАВНИВАНИЕ ИЗОБРАЖЕНИЯ")
    print("-" * 40)
    
    img_aligned = deskew_image(img)
    
    if save_images:
        aligned_path = os.path.join(output_dir, f"{img_name}_aligned.jpg")
        cv2.imwrite(aligned_path, img_aligned)
        print(f"✓ Выравненное изображение сохранено: {aligned_path}")
    
    # ============ ЭТАП 2: ИЗВЛЕЧЕНИЕ БЛОКОВ ============
    print("\n📦 ЭТАП 2: ИЗВЛЕЧЕНИЕ ПРЯМОУГОЛЬНЫХ БЛОКОВ")
    print("-" * 40)
    
    # Конвертация в оттенки серого
    if len(img_aligned.shape) == 3:
        gray = cv2.cvtColor(img_aligned, cv2.COLOR_BGR2GRAY)
    else:
        gray = img_aligned.copy()
    
    # Сохраняем цветную версию для отрисовки
    if len(img_aligned.shape) == 3:
        img_color = img_aligned.copy()
    else:
        img_color = cv2.cvtColor(img_aligned, cv2.COLOR_GRAY2BGR)
    
    # ============ СОХРАНЯЕМ ВСЕ ПРОМЕЖУТОЧНЫЕ ЭТАПЫ ============
    debug_images = []
    debug_names = []
    debug_paths = []
    
    # 1. Исходное изображение (серый)
    debug_images.append(gray.copy())
    debug_names.append("1. Исходное изображение (серый)")
    debug_paths.append("01_original_gray")
    
    # 2. Дилатация
    footprint = np.ones((dilation_size, dilation_size), dtype=bool)
    dilated = scipy_morph.grey_dilation(gray, footprint=footprint)
    debug_images.append(dilated.copy())
    debug_names.append(f"2. Дилатация (kernel={dilation_size}x{dilation_size})")
    debug_paths.append(f"02_dilation_kernel_{dilation_size}")
    
    # 3. Морфологический градиент
    gradient = dilated.astype(np.int32) - gray.astype(np.int32)
    gradient_display = np.clip(gradient, 0, 255).astype(np.uint8)
    debug_images.append(gradient_display)
    debug_names.append("3. Морфологический градиент")
    debug_paths.append("03_morphological_gradient")
    
    # 4. Статистика градиента
    mean_val = gradient.mean()
    std_val = gradient.std()
    threshold = mean_val + (std_val * std_multiplier)
    
    print(f"   Статистика градиента:")
    print(f"   - Среднее значение: {mean_val:.2f}")
    print(f"   - Стандартное отклонение: {std_val:.2f}")
    print(f"   - Порог (mean + {std_multiplier}*std): {threshold:.2f}")
    
    # 5. Бинаризация
    binary = gradient.copy()
    binary[binary < threshold] = 0
    binary[binary >= threshold] = 1
    binary_display = (binary * 255).astype(np.uint8)
    debug_images.append(binary_display)
    debug_names.append(f"4. Бинаризация (порог={threshold:.2f})")
    debug_paths.append(f"04_binary_threshold_{threshold:.2f}")
    
    # 6. Инвертированная бинарная для контуров (опционально)
    binary_inv = cv2.bitwise_not(binary_display)
    debug_images.append(binary_inv)
    debug_names.append("5. Инвертированная бинарная")
    debug_paths.append("05_binary_inverted")
    
    # 7. Морфологические операции для улучшения
    kernel = np.ones((3, 3), np.uint8)
    binary_closed = cv2.morphologyEx(binary_display, cv2.MORPH_CLOSE, kernel)
    debug_images.append(binary_closed)
    debug_names.append("6. Морфологическое закрытие (3x3)")
    debug_paths.append("06_morphological_close")
    
    # 8. Связные компоненты
    lbl, numcc = label(binary)
    
    # Визуализация меток (каждая компонента своим цветом)
    lbl_colored = np.zeros((*lbl.shape, 3), dtype=np.uint8)
    component_sizes = []
    for i in range(1, numcc + 1):
        color = (random.randint(50, 255), random.randint(50, 255), random.randint(50, 255))
        lbl_colored[lbl == i] = color
        component_sizes.append(np.sum(lbl == i))
    
    debug_images.append(lbl_colored)
    debug_names.append(f"7. Связные компоненты (всего: {numcc})")
    debug_paths.append(f"07_connected_components_{numcc}")
    
    print(f"\n   Связные компоненты:")
    print(f"   - Всего найдено: {numcc}")
    print(f"   - Минимальный размер: {min_size} пикселей")
    
    # 9. Фильтрация компонентов
    boxes_coords = []
    boxes_img = img_color.copy()
    components_filtered = np.zeros_like(lbl_colored)
    filtered_count = 0
    rejected_count = 0
    
    for i in range(1, numcc + 1):
        py, px = np.nonzero(lbl == i)
        component_size = len(py)
        
        # Если компонента слишком маленькая, игнорируем
        if component_size < min_size:
            rejected_count += 1
            continue
        
        filtered_count += 1
        components_filtered[lbl == i] = lbl_colored[lbl == i]
        
        xmin, xmax, ymin, ymax = px.min(), px.max(), py.min(), py.max()
        x, y, w, h = xmin, ymin, (xmax - xmin), (ymax - ymin)
        
        boxes_coords.append((x, y, w, h))
        
        # Рисуем прямоугольник с случайным цветом
        color = (random.randint(100, 255), random.randint(100, 255), random.randint(100, 255))
        cv2.rectangle(boxes_img, (x, y), (x + w, y + h), color, 2)
        
        # Добавляем центр
        cx = x + w // 2
        cy = y + h // 2
        cv2.circle(boxes_img, (cx, cy), 4, (0, 255, 255), -1)
        
        # Добавляем номер
        cv2.putText(boxes_img, str(filtered_count), (x, y-5), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)
        
        # Выводим информацию о компоненте
        print(f"   Компонента {filtered_count}: размер={component_size}, x={x}, y={y}, w={w}, h={h}")
    
    # Сохраняем визуализации
    debug_images.append(components_filtered)
    debug_names.append(f"8. Отфильтрованные компоненты ({filtered_count} из {numcc})")
    debug_paths.append(f"08_filtered_components_{filtered_count}")
    
    debug_images.append(boxes_img)
    debug_names.append(f"9. Найденные прямоугольники ({filtered_count})")
    debug_paths.append(f"09_final_boxes_{filtered_count}")
    
    # ============ СОХРАНЕНИЕ ВСЕХ ДЕБАГ ИЗОБРАЖЕНИЙ ============
    if save_images:
        print(f"\n💾 Сохранение дебаг изображений в: {debug_dir}")
        for debug_img, debug_path, debug_name in zip(debug_images, debug_paths, debug_names):
            save_path = os.path.join(debug_dir, f"{img_name}_{debug_path}.jpg")
            cv2.imwrite(save_path, debug_img)
        print(f"✓ Сохранено {len(debug_images)} дебаг изображений")
    
    # Сохраняем финальный результат
    if save_images:
        boxes_path = os.path.join(output_dir, f"{img_name}_boxes.jpg")
        cv2.imwrite(boxes_path, boxes_img)
        print(f"✓ Результат с боксами сохранен: {boxes_path}")
    
    # ============ СТАТИСТИКА ============
    print("\n" + "="*60)
    print("📊 СТАТИСТИКА ОБРАБОТКИ:")
    print(f"   - Всего компонент (блоков) найдено: {numcc}")
    print(f"   - Отфильтровано (подходят по размеру): {filtered_count}")
    print(f"   - Отклонено (слишком маленькие): {rejected_count}")
    print(f"   - Параметры: Dilation={dilation_size}, Std Multiplier={std_multiplier:.2f}, MinSize={min_size}")
    print("="*60)
    
    if boxes_coords:
        print("\n📝 НАЙДЕННЫЕ ПРЯМОУГОЛЬНИКИ:")
        for i, (x, y, w, h) in enumerate(boxes_coords):
            area = w * h
            print(f"   {i+1}: x={x:>4}, y={y:>4}, w={w:>4}, h={h:>4}, площадь={area:>7}")
    
    print("="*60)
    
    # ============ ВИЗУАЛИЗАЦИЯ ============
    if show_plots:
        # Показываем все этапы обработки
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
        
        # Скрываем неиспользуемые подграфики
        for i in range(n_images, len(axes)):
            axes[i].axis('off')
        
        plt.tight_layout()
        
        if save_images:
            summary_path = os.path.join(output_dir, f"{img_name}_debug_summary.png")
            fig.savefig(summary_path, dpi=150, bbox_inches='tight')
            print(f"✓ Сводка дебаг изображений сохранена: {summary_path}")
        
        plt.show()
    
    return boxes_img, boxes_coords

# ==================== ПАРТИЙНАЯ ОБРАБОТКА ============
def batch_process(input_dir, output_dir="./output/", 
                  dilation_size=3, std_multiplier=1.0, min_size=200,
                  show_plots=False, save_images=True):
    """
    Обработка всех изображений в папке
    """
    input_path = Path(input_dir)
    if not input_path.exists():
        print(f"Ошибка: папка {input_dir} не существует")
        return
    
    # Поддерживаемые форматы
    extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff'}
    image_files = [f for f in input_path.iterdir() if f.suffix.lower() in extensions]
    
    if not image_files:
        print(f"В папке {input_dir} нет изображений")
        return
    
    print(f"Найдено {len(image_files)} изображений для обработки")
    print("="*60)
    
    results = {}
    for img_file in image_files:
        print(f"\n🔄 Обработка: {img_file.name}")
        boxes_img, boxes_coords = process_image_with_debug(
            str(img_file),
            dilation_size=dilation_size,
            std_multiplier=std_multiplier,
            min_size=min_size,
            output_dir=output_dir,
            show_plots=show_plots,
            save_images=save_images
        )
        results[img_file.name] = {
            'boxes_img': boxes_img,
            'boxes_coords': boxes_coords,
            'num_boxes': len(boxes_coords) if boxes_coords else 0
        }
    
    # Общая статистика
    print("\n" + "="*60)
    print("📊 ОБЩАЯ СТАТИСТИКА ОБРАБОТКИ:")
    for name, data in results.items():
        print(f"   {name}: найдено {data['num_boxes']} блоков")
    print("="*60)
    
    return results

# ==================== ЗАПУСК ============
if __name__ == "__main__":
    # Параметры обработки
    input_image = "/media/vadim/1TB_SSD/my_github/computer-vision-document-table-parser/input_images/1.jpg"
    
    # Настройки для извлечения блоков
    dilation_size = 7       # Размер ядра дилатации (должен быть нечетным: 1,3,5,7,9)
    std_multiplier = 1.0    # Множитель стандартного отклонения для порога
    min_size = 200          # Минимальный размер блока в пикселях
    
    clean_output_folder("./output_finish/")

    # Запуск обработки одного изображения
    result_img, boxes = process_image_with_debug(
        img_path=input_image,
        dilation_size=dilation_size,
        std_multiplier=std_multiplier,
        min_size=min_size,
        output_dir="./output_finish/",
        show_plots=True,      # Показать все этапы обработки
        save_images=True      # Сохранить все дебаг изображения
    )
    
    # Для пакетной обработки раскомментируйте:
    # batch_process(
    #     input_dir="/media/vadim/1TB_SSD/my_github/computer-vision-document-table-parser/input_images/",
    #     output_dir="./output/",
    #     dilation_size=7,
    #     std_multiplier=1.0,
    #     min_size=200,
    #     show_plots=False,
    #     save_images=True
    # )