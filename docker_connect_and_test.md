
vadim@vadim-ms-7c39:/media/vadim/1TB_SSD/my_github/computer-vision-document-table-parser$ docker compose run --rm app /bin/bash
WARN[0000] /media/vadim/1TB_SSD/my_github/computer-vision-document-table-parser/docker-compose.yml: `version` is obsolete 
root@80f80ef38e9a:/app# ды
bash: ды: command not found
root@80f80ef38e9a:/app# ls
01_rotate_image.py                02_rotate_and_parse.py        input_images   raw_dir           test_data
02_rotate_and_ocr_parse_debug.py  02_rotate_and_parse_debug.py  output_finish  requirements.txt
root@80f80ef38e9a:/app# cd r
raw_dir/          requirements.txt  
root@80f80ef38e9a:/app# cd raw_dir
root@80f80ef38e9a:/app/raw_dir# ls
01_rotate_image.py                                  03_updated_stoverf.ipynb  environment-min.yml
01_test_new_method.ipynb                            04_mser_opencv.ipynb      input_images
02_rotate_and_ocr_parse_debug.py                    Dockerfile                inter_processing
02_rotate_and_parse.py                              MY_SUMMARY.md             output
02_rotate_and_parse_debug.py                        README.md                 output_finish
02_rotate_image.ipynb                               RUN.md                    prepare_for_docker.md
02_test_hought_lines.ipynb                          TODO.md                   requirements.txt
03_brocken_counters__retr_comp.ipynb                doc                       result_with_tables.jpg
03_broken_counters_from_stackoverflow.ipynb         doc.md                    test_data
03_broken_counters_from_stackoverflow__debug.ipynb  docker-compose.yml
03_test_find_broken_counters.ipynb                  document_with_tables.jpg
root@80f80ef38e9a:/app/raw_dir# python 02_rotate_and_ocr_parse_debug.py 
/usr/local/lib/python3.10/site-packages/fuzzywuzzy/fuzz.py:11: UserWarning: Using slow pure-python SequenceMatcher. Install python-Levenshtein to remove this warning
  warnings.warn('Using slow pure-python SequenceMatcher. Install python-Levenshtein to remove this warning')
✓ Папка ./output_finish/ очищена
Обработка изображения: ./input_images/4.jpg
============================================================

📐 ЭТАП 1: ВЫРАВНИВАНИЕ ИЗОБРАЖЕНИЯ
----------------------------------------
Обнаружен угол наклона: 0.18 градусов
✓ Выравненное изображение сохранено: ./output_finish/4_aligned.jpg

📦 ЭТАП 2: ИЗВЛЕЧЕНИЕ ПРЯМОУГОЛЬНЫХ БЛОКОВ
----------------------------------------
/app/raw_dir/02_rotate_and_ocr_parse_debug.py:371: DeprecationWarning: Please import `grey_dilation` from the `scipy.ndimage` namespace; the `scipy.ndimage.morphology` namespace is deprecated and will be removed in SciPy 2.0.0.
  dilated = scipy_morph.grey_dilation(gray, footprint=footprint)
   Статистика градиента:
   - Среднее значение: 23.37
   - Стандартное отклонение: 55.60
   - Порог (mean + 1.0*std): 78.97

   Связные компоненты:
   - Всего найдено: 349
   - Минимальный размер: 500 пикселей
   Компонента 1: размер=4030, x=378, y=58, w=694, h=58
   Компонента 2: размер=896, x=229, y=131, w=173, h=30
   Компонента 3: размер=691, x=622, y=131, w=132, h=29
   Компонента 4: размер=691, x=968, y=131, w=103, h=30
   Компонента 5: размер=3713, x=361, y=175, w=711, h=35
   Компонента 6: размер=3636, x=361, y=214, w=711, h=35
   Компонента 7: размер=3114, x=361, y=253, w=711, h=35
   Компонента 8: размер=4019, x=361, y=298, w=711, h=35
   Компонента 9: размер=3717, x=361, y=370, w=711, h=36
   Компонента 10: размер=4112, x=361, y=408, w=710, h=37
   Компонента 11: размер=4083, x=361, y=453, w=710, h=37

💾 Сохранение дебаг изображений в: ./output_finish/debug
✓ Сохранено 9 дебаг изображений
✓ Результат с боксами сохранен: ./output_finish/4_boxes.jpg

📝 ЭТАП 3: РАСПОЗНАВАНИЕ ТЕКСТА (EasyOCR)
----------------------------------------

🔍 Инициализация EasyOCR (русский + английский)...
Downloading detection model, please wait. This may take several minutes depending upon your network connection.
Downloading recognition model, please wait. This may take several minutes depending upon your network connection.

📝 Распознавание текста в 11 боксах...
--------------------------------------------------
/usr/local/lib/python3.10/site-packages/torch/utils/data/dataloader.py:775: UserWarning: 'pin_memory' argument is set as true but no accelerator is found, then device pinned memory won't be used.
  super().__init__(loader)
   Бокс 1: 'Регламента рассмотрения проектных решений Главным архитектором L МОсквЫ' (conf=0.747, фрагментов=3)
   Бокс 2: '05.07.99' (conf=0.885, фрагментов=1)
   Бокс 3: '23' (conf=1.000, фрагментов=1)
   Бокс 4: '86' (conf=1.000, фрагментов=1)
   Бокс 5: 'ТЭО строительства мини-кондитерскорго цеха' (conf=0.838, фрагментов=1)
   Бокс 6: 'Ширнов ЮА ; Тиморев Л Г' (conf=0.896, фрагментов=4)
   Бокс 7: 'институт Мосстройпроект"' (conf=0.995, фрагментов=2)
   Бокс 8: 'ОAО 'Инвестпроект' (conf=0.744, фрагментов=2)
   Бокс 9: 'Попов В.П' (conf=0.812, фрагментов=2)
   Бокс 10: текст не найден
   Бокс 11: текст не найден
--------------------------------------------------
✓ Результаты сохранены в: ./output_finish/ocr_results.json
⚠️ Шрифт с поддержкой кириллицы не найден, используется стандартный
✓ Визуализация сохранена: ./output_finish/4_ocr_visualization.jpg
✓ Маппированные результаты сохранены в: ./output_finish/4_mapped_results.json

❌ Тест 4: НЕ ПРОЙДЕН
   ОШИБКИ:
   • Отсутствует поле: Рассмотрение на рабочей комиссии

   📊 СТАТИСТИКА:
   • Всего полей: 12
   • Совпадает: 11
   • Не совпадает: 0
   • Средняя схожесть: 99.1%

💾 Результат сохранен: ./test_data/output/4_result.json

📊 СТАТИСТИКА OCR:
   - Всего боксов: 11
   - Распознано текста: 9
   - Текст не найден: 2
   - Пустые области: 0
   - Средняя уверенность: 0.880

============================================================
📊 ОБЩАЯ СТАТИСТИКА:
   - Найдено блоков: 11
   - Отклонено блоков: 338
   - Параметры: Dilation=7, Std Multiplier=1.00, MinSize=500
   - Распознано текста: 9/11
============================================================
✓ Сводка дебаг изображений сохранена: ./output_finish/4_debug_summary.png

✨ ОБРАБОТКА ЗАВЕРШЕНА ✨
📁 Результаты сохранены в папке: ./output_finish/
   - ocr_results.json - результаты распознавания
   - 4_ocr_visualization.jpg - визуализация с текстом
   - 4_aligned.jpg - выравненное изображение
   - 4_boxes.jpg - найденные боксы
   - debug/ - все промежуточные этапы
root@80f80ef38e9a:/app/raw_dir# 