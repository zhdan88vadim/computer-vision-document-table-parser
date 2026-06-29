FROM python:3.10-slim

# Устанавливаем системные зависимости для OpenCV и EasyOCR
RUN apt-get update && apt-get install -y \
    libgl1-mesa-glx \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libgomp1 \
    wget \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Копируем requirements.txt и устанавливаем Python-пакеты
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Копируем ваш скрипт
COPY 02_rotate_and_ocr_parse_debug.py .

# Команда запуска
CMD ["python", "02_rotate_and_ocr_parse_debug.py"]