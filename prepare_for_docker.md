source /mnt/ntfs/learn_ML/opencv_torch_env/bin/activate


Чтобы убрать специфичную для вашей платформы информацию о сборке (build), используйте флаг --no-builds :

bash
conda env export --no-builds > environment.yml
Экспорт только явно установленных вами пакетов Conda: Этот вариант создает более "чистый" файл, содержащий только те пакеты, которые вы установили напрямую, без их зависимостей .

bash
conda env export --from-history > environment-min.yml



pip freeze > requirements.txt



pip show fuzzywuzzy




# with files
find . -not -path "*.git*" | sed -e "s/[^-][^\/]*\// |/g" -e "s/|\([^ ]\)/|-\1/"


# only folders
find . -type d -not -path "*.git*" | sed -e "s/[^-][^\/]*\// |/g" -e "s/|\([^ ]\)/|-\1/"











# update docker compose


rm -f ~/.docker/cli-plugins/docker-compose
# Remove all possible docker-compose locations
sudo rm -f /usr/local/lib/docker/cli-plugins/docker-compose
sudo rm -f /usr/lib/docker/cli-plugins/docker-compose
sudo rm -f /usr/libexec/docker/cli-plugins/docker-compose
sudo rm -f ~/.docker/cli-plugins/docker-compose
sudo rm -f /usr/local/bin/docker-compose
sudo rm -f /usr/bin/docker-compose

# Also remove any old docker-compose from PATH
sudo rm -f /snap/bin/docker-compose 2>/dev/null

# Now install the new version
sudo mkdir -p /usr/local/lib/docker/cli-plugins
sudo curl -SL "https://github.com/docker/compose/releases/download/v2.27.1/docker-compose-$(uname -s)-$(uname -m)" -o /usr/local/lib/docker/cli-plugins/docker-compose
sudo chmod +x /usr/local/lib/docker/cli-plugins/docker-compose

# Check which one is being used
docker compose version











vadim@vadim-ms-7c39:/media/vadim/1TB_SSD/my_github/computer-vision-document-table-parser$ docker compose up
WARN[0000] /media/vadim/1TB_SSD/my_github/computer-vision-document-table-parser/docker-compose.yml: `version` is obsolete 
[+] Running 2/0
 ✔ Network computer-vision-document-table-parser_default  Created                                                                 0.0s 
 ✔ Container container-comp-vision-table-parser           Created                                                                 0.0s 
Attaching to container-comp-vision-table-parser
container-comp-vision-table-parser  | /usr/local/lib/python3.10/site-packages/fuzzywuzzy/fuzz.py:11: UserWarning: Using slow pure-python SequenceMatcher. Install python-Levenshtein to remove this warning
container-comp-vision-table-parser  |   warnings.warn('Using slow pure-python SequenceMatcher. Install python-Levenshtein to remove this warning')
container-comp-vision-table-parser  | [ WARN:0@2.934] global loadsave.cpp:278 findDecoder imread_('/media/vadim/1TB_SSD/my_github/computer-vision-document-table-parser/input_images/3.jpg'): can't open/read file: check file path/integrity
container-comp-vision-table-parser  | ✓ Папка ./output_finish/ очищена
container-comp-vision-table-parser  | Ошибка: не удалось загрузить изображение /media/vadim/1TB_SSD/my_github/computer-vision-document-table-parser/input_images/3.jpg
container-comp-vision-table-parser  | 
container-comp-vision-table-parser  | ✨ ОБРАБОТКА ЗАВЕРШЕНА ✨
container-comp-vision-table-parser  | 📁 Результаты сохранены в папке: ./output_finish/
container-comp-vision-table-parser  |    - ocr_results.json - результаты распознавания
container-comp-vision-table-parser  |    - ocr_visualization.jpg - визуализация с текстом
container-comp-vision-table-parser  |    - 3_aligned.jpg - выравненное изображение
container-comp-vision-table-parser  |    - 3_boxes.jpg - найденные боксы
container-comp-vision-table-parser  |    - debug/ - все промежуточные этапы
container-comp-vision-table-parser exited with code 0
vadim@vadim-ms-7c39:/media/vadim/1TB_SSD/my_github/computer-vision-document-table-parser$ 



