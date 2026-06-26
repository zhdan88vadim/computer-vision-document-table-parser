Если нужно найти отдельные ячейки, то метод компонент находит хорошо, даже с небольшими разрывами.

Если нужно поделить таблицу на ячейки, то HoughLinesP искал хорошо на судоку.
и поиск линий и затем мерж тоже вроде ок


# Вертикальные линии
v_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, k_size))
v_lines = cv2.dilate(cv2.erode(img_bin, v_kernel, iterations=2), v_kernel, iterations=2)

# Горизонтальные линии
h_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (k_size, 1))
h_lines = cv2.dilate(cv2.erode(img_bin, h_kernel, iterations=2), h_kernel, iterations=2)

# Объединение
combined = cv2.addWeighted(v_lines, 0.5, h_lines, 0.5, 0)