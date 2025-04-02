import pandas as pd
from deep_translator import GoogleTranslator
from tqdm import tqdm
import time
import logging
import os

# Настройка логирования
logging.basicConfig(filename='translate_errors.log', level=logging.ERROR, format='%(asctime)s - %(message)s')

# Файл исходный и файл для сохранения
file_path = "train.csv"
output_path = "ru_translated.csv"

# Загрузка файла (если перезапуск — берем сохранённый)
if os.path.exists(output_path):
    df = pd.read_csv(output_path)
    print("Продолжение с последнего места из 'ru_translated.csv'")
else:
    df = pd.read_csv(file_path)
    df['russian_text'] = ""

# Функция для перевода
def translate_text(text):
    try:
        return GoogleTranslator(source='en', target='ru').translate(text)
    except Exception as e:
        logging.error(f"Ошибка при переводе текста: {text} — {e}")
        return ""

# Перебор строк с прогресс-баром
for idx in tqdm(range(len(df)), desc="Перевод строк", ncols=100):
    eng_text = df.loc[idx, 'english_text']
    rus_text = df.loc[idx, 'russian_text']

    if pd.notna(rus_text) and str(rus_text).strip() != "":
        continue  # Пропускаем, если перевод уже есть

    print(f"\nПеревод [{idx+1}/{len(df)}]: {eng_text}")
    translated = translate_text(eng_text)
    if translated:
        df.loc[idx, 'russian_text'] = translated
    else:
        df.loc[idx, 'russian_text'] = ""  # На всякий случай явно сохраняем пустое

    df.to_csv(output_path, index=False)
    time.sleep(1.5)  # Задержка, чтобы не перегрузить API

print("✅ Перевод завершён. Сохранено в 'ru_translated.csv'")
