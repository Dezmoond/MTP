from transformers import PreTrainedTokenizerFast, AutoModelForCausalLM, Trainer, TrainingArguments
from datasets import load_dataset
import torch
import json
import torch.optim as optim
device = torch.device("cuda")

torch.cuda.empty_cache()
# Загрузка модели и токенизатора
model_path = "D:/anaconda3/envs/Lama5/Lama5/Llama-3.2-1B-Instruct"
tokenizer = PreTrainedTokenizerFast.from_pretrained(model_path)
model = AutoModelForCausalLM.from_pretrained(model_path).to(device)

# Переводим модель на Gpu

model.to(device)

# Заморозить первые 5 слоев
for name, param in model.named_parameters():
    if "transformer.h" in name and int(name.split('.')[1]) < 5:
        param.requires_grad = False

# Загрузка данных из файла data3.txt в формате JSON
file_path = "mtp/llamatrain1.txt"  #Путь к вашему файлу

# Открываем и читаем JSON данные
with open(file_path, "r", encoding="utf-8") as f:
    dataset_raw = json.load(f)

# Проверим структуру данных
print(dataset_raw[:2])  # Выведем первые два примера для проверки

# Преобразуем данные в формат, который будет использоваться для создания датасета
from datasets import Dataset

# Создаем Dataset из словарей
dataset = Dataset.from_dict({
    "prompt": [entry["prompt"] for entry in dataset_raw],
    "response": [entry["response"] for entry in dataset_raw]
})

# Проверим структуру данных
print(dataset)

# Токенизация данных
tokenizer.pad_token = tokenizer.eos_token  # Используем токен EOS как токен для паддинга

def tokenize_function(examples):
    # Формируем текст в формате "prompt: {вопрос} response: {ответ}"
    text = [f"prompt: {examples['prompt'][i]} response: {examples['response'][i]}" for i in range(len(examples["prompt"]))]

    # Токенизация с паддингом и обрезкой
    encodings = tokenizer(text, padding="max_length", truncation=True, max_length=128)

    # Добавляем метки в правильном формате
    encodings["labels"] = encodings["input_ids"]

    return encodings

# Применяем токенизацию
tokenized_dataset = dataset.map(tokenize_function, batched=True)

# Разделение на тренировочные и валидационные данные
train_dataset = tokenized_dataset
eval_dataset = tokenized_dataset  # Для упрощения используем одни и те же данные для тренировки и оценки

# Настройки тренировки
training_args = TrainingArguments(
    output_dir="./results",
    learning_rate=2e-5,
    per_device_train_batch_size=15,
    weight_decay=0.2,
    num_train_epochs=1,
    logging_dir='./logs',
    evaluation_strategy="steps",  # Оценка после определенного количества шагов
    eval_steps=1000,  # Оценка после 500 шагов
    fp16=True,  # Включение смешанной точности
)
#gradient_accumulation_steps=2,  # Для накопления градиентов божно без этого и 20 батч

# Создание тренера
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
    optimizers=(optim.AdamW(model.parameters(), lr=2e-5), None),
)

# Запуск тренировки
trainer.train()

# Сохранение модели и токенизатора
model.save_pretrained("/MTP8")
tokenizer.save_pretrained("/MTP8")
