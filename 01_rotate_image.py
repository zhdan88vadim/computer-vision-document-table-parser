import cv2
import numpy as np

def deskew_image(image_path, output_path):
    # 1. Загрузка изображения
    img = cv2.imread(image_path)
    if img is None:
        print("Ошибка: не удалось загрузить изображение")
        return

    # 2. Предобработка: перевод в оттенки серого и размытие (убираем шум)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (5, 5), 0)

    # 3. Бинаризация (делаем четкие черно-белые границы)
    # Используем адаптивный порог, так как документ может быть освещен неравномерно
    thresh = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                   cv2.THRESH_BINARY_INV, 11, 2)

    # 4. Поиск контуров
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if not contours:
        print("Контуры не найдены")
        return

    # 5. Находим самый большой контур (скорее всего это сама рамка документа)
    largest_contour = max(contours, key=cv2.contourArea)

    # 6. Аппроксимируем контур до прямоугольника (находим углы рамки)
    # Метод minAreaRect находит минимальный описанный прямоугольник, который идеально подходит для выравнивания
    rect = cv2.minAreaRect(largest_contour)
    
    # Угол наклона (в градусах)
    angle = rect[2]
    
    # Логика OpenCV по определению угла: если ширина меньше высоты, угол отрицательный.
    # Нам нужен угол для вращения относительно центра.
    if angle < -45:
        angle = 90 + angle
    
    # Если угол слишком мал, выравнивать не нужно (чтобы не дрожать на ровных фото)
    if abs(angle) < 0.5:
        print("Изображение уже ровное, угол наклона:", angle)
        cv2.imwrite(output_path, img)
        return

    print(f"Обнаружен угол наклона: {angle:.2f} градусов")

    # 7. Получаем размеры изображения и находим центр
    (h, w) = img.shape[:2]
    center = (w // 2, h // 2)

    # 8. Создаем матрицу поворота (M)
    M = cv2.getRotationMatrix2D(center, angle, 1.0)

    # 9. Вычисляем новые границы, чтобы картинка не обрезалась сильно
    # Если вращать просто так, края документа могут отрезаться.
    # Чтобы сохранить всё, нужно растянуть матрицу:
    cos = np.abs(M[0, 0])
    sin = np.abs(M[0, 1])
    new_w = int((h * sin) + (w * cos))
    new_h = int((h * cos) + (w * sin))
    
    # Корректируем матрицу для смещения центра при новом размере
    M[0, 2] += (new_w / 2) - center[0]
    M[1, 2] += (new_h / 2) - center[1]

    # 10. Применяем поворот. BORDER_CONSTANT заливает фон белым (или черным), чтобы не было рваных краев
    rotated = cv2.warpAffine(img, M, (new_w, new_h), borderMode=cv2.BORDER_CONSTANT, borderValue=(255, 255, 255))

    # 11. Сохраняем результат
    cv2.imwrite(output_path, rotated)
    print(f"Изображение сохранено как: {output_path}")

# Запуск
deskew_image('/media/vadim/1TB_SSD/my_github/computer-vision-document-table-parser/input_images/1.jpg', 'deskewed_output.jpg')