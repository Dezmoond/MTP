import torch
import json
import re
import pandas as pd
from collections import defaultdict
from transformers import PreTrainedTokenizerFast, AutoModelForCausalLM

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print('Using device:', device)

# Путь к модели
model_path = "model/MTP7"

# Загрузка токенизатора и модели
tokenizer = PreTrainedTokenizerFast.from_pretrained(model_path)
model = AutoModelForCausalLM.from_pretrained(model_path).to(device)

# Проверка pad_token_id и его настройка, если необходимо
if isinstance(model.config.pad_token_id, list):
    model.config.pad_token_id = model.config.pad_token_id[0]
if model.config.pad_token_id is None:
    model.config.pad_token_id = model.config.eos_token_id[0] if isinstance(model.config.eos_token_id, list) else model.config.eos_token_id

# Функция для генерации ответа
def generate_response(input_text):
    system_prompt = 'Вы эксперт, классифицирующий товар по категориям. Ответ должен содержать только номер категории без слов и пояснений. Категорий всего 84 не более.'
    prompt = f"{system_prompt}\nВопрос: {input_text}\nОтвет:"
    inputs = tokenizer(prompt, return_tensors="pt").to(device)

    # Генерация ответа
    output = model.generate(
        **inputs,
        max_length=200,
        num_return_sequences=1,
        do_sample=True,
        top_p=0.05,
        temperature=0.1,
        top_k=10,
        pad_token_id=model.config.pad_token_id
    )

    # Декодирование ответа
    response_text = tokenizer.decode(output[0], skip_special_tokens=True)
    return response_text.strip()

# Функция для извлечения последнего числа из ответа модели
def extract_last_number(text):
    matches = re.findall(r'\d+', text)  # Находим все числа в тексте
    return matches[-1] if matches else None  # Берём последнее найденное число

# Загрузка категорий из CSV файла
category_df = pd.read_csv('mtp/category.csv', delimiter=';', encoding='utf-8')
category_dict = dict(zip(category_df['number_category'], category_df['name_category']))

# Чтение данных
with open('mtp/llamatest6.txt', 'r', encoding='utf-8') as file:
    data = json.load(file)

# Подсчет правильных/неправильных ответов
correct_answers = 0
total_questions = len(data)

# Статистика ошибок по категориям
category_errors = defaultdict(lambda: {"total": 0, "errors": 0})

for entry in data:
    prompt = entry["prompt"]
    expected_response = str(entry["response"])  # Приводим к строке

    # Генерация ответа
    generated_response = generate_response(prompt)

    # Извлекаем число
    generated_number = extract_last_number(generated_response)

    # Получаем название категории
    category_name = category_dict.get(int(generated_number), "Неизвестная категория")

    # Обновляем статистику по категории
    category_errors[expected_response]["total"] += 1
    if generated_number != expected_response:
        category_errors[expected_response]["errors"] += 1

    # Подсчет правильных ответов
    if generated_number == expected_response:
        correct_answers += 1

    # Выводим подробности
    print(f"Продукт: {prompt}")
    print(f"Ответ модели: {generated_response}")
    print(f"Выделенный ответ: {generated_number}")
    print(f"Правильный ответ: {expected_response}")
    print(f"Определенная категория: {category_name}")
    print("-" * 50)

# Статистика
accuracy = (correct_answers / total_questions) * 100
print("\n📊 Общая статистика:")
print(f"Процент правильных ответов: {accuracy:.2f}%\n")

# Вывод категорий с процентом ошибок
print("📉 Ошибки по категориям:")
sorted_errors = sorted(category_errors.items(), key=lambda x: x[1]["errors"], reverse=True)

for category, stats in sorted_errors:
    total = stats["total"]
    errors = stats["errors"]
    error_rate = (errors / total) * 100 if total > 0 else 0
    print(f"Категория {category}: ошибок {errors} из {total} ({error_rate:.2f}%)")
