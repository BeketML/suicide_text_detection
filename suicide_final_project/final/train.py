import os
import pandas as pd
import torch
from torch.utils.data import Dataset
from transformers import (
    XLMRobertaTokenizer,
    XLMRobertaForSequenceClassification,
    TrainingArguments,
    Trainer,
    EarlyStoppingCallback,
    DataCollatorWithPadding
)
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
import numpy as np
from typing import List, Tuple
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class SuicideTextDataset(Dataset):
    """Dataset для классификации суицидальных текстов"""
    
    def __init__(
        self, 
        texts: List[str], 
        labels: List[int], 
        tokenizer, 
        max_length: int = 512
    ):
        """
        Args:
            texts: Список текстов
            labels: Список меток (0 или 1)
            tokenizer: Токенизатор XLM-RoBERTa
            max_length: Максимальная длина последовательности
        """
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length
    
    def __len__(self):
        return len(self.texts)
    
    def __getitem__(self, idx):
        text = str(self.texts[idx])
        label = self.labels[idx]
        
        encoding = self.tokenizer(
            text,
            truncation=True,
            max_length=self.max_length,
            return_tensors=None
        )
        
        return {
            'input_ids': encoding['input_ids'],
            'attention_mask': encoding['attention_mask'],
            'labels': torch.tensor(label, dtype=torch.long)
        }


def compute_metrics(eval_pred):
    """Вычисляет метрики для оценки модели"""
    predictions, labels = eval_pred
    predictions = np.argmax(predictions, axis=1)
    
    precision, recall, f1, _ = precision_recall_fscore_support(
        labels, predictions, average='weighted', zero_division=0
    )
    accuracy = accuracy_score(labels, predictions)
    
    return {
        'accuracy': accuracy,
        'f1': f1,
        'precision': precision,
        'recall': recall
    }


def load_and_prepare_data(
    csv_path: str,
    text_column: str = 'clean_text',
    label_column: str = 'label',
    test_size: float = 0.1,
    val_size: float = 0.1,
    random_state: int = 42
) -> Tuple[
    List[str], List[str], List[str],  # train/val/test texts
    List[int], List[int], List[int]   # train/val/test labels
]:
    """
    Загружает и разделяет данные на train/val/test
    
    Returns:
        train_texts, val_texts, test_texts, 
        train_labels, val_labels, test_labels
    """
    logger.info(f"Загрузка данных из {csv_path}")
    df = pd.read_csv(csv_path)
    
    logger.info(f"Размер датасета: {len(df)} строк")
    logger.info(f"Распределение меток:\n{df[label_column].value_counts()}")
    
    # Используем только clean_text и label
    texts = df[text_column].fillna('').astype(str).tolist()
    labels = df[label_column].astype(int).tolist()
    
    train_texts, temp_texts, train_labels, temp_labels = train_test_split(
        texts, labels, 
        test_size=(test_size + val_size), 
        random_state=random_state, 
        stratify=labels
    )
    
    val_texts, test_texts, val_labels, test_labels = train_test_split(
        temp_texts, temp_labels,
        test_size=test_size / (test_size + val_size), 
        random_state=random_state, 
        stratify=temp_labels
    )
    
    logger.info(f"Train: {len(train_texts)}, Val: {len(val_texts)}, Test: {len(test_texts)}")
    
    return (
        train_texts, val_texts, test_texts,
        train_labels, val_labels, test_labels
    )


def train_model(
    model_name: str = "FacebookAI/xlm-roberta-base",
    csv_path: str = None,
    output_dir: str = "./results",
    num_epochs: int = 3,
    batch_size: int = 16,
    learning_rate: float = 2e-5,
    max_length: int = 512,
    warmup_steps: int = 500,
    weight_decay: float = 0.01,
    save_steps: int = 500,
    eval_steps: int = 500,
    logging_steps: int = 100,
    text_column: str = 'clean_text',
    label_column: str = 'label'
):
    """
    Основная функция для обучения модели XLM-RoBERTa-base
    
    Args:
        model_name: Название модели из HuggingFace (FacebookAI/xlm-roberta-base)
        csv_path: Путь к CSV файлу с данными
        output_dir: Директория для сохранения результатов
        num_epochs: Количество эпох обучения
        batch_size: Размер батча
        learning_rate: Скорость обучения
        max_length: Максимальная длина последовательности
        warmup_steps: Количество шагов для warmup
        weight_decay: Вес для регуляризации
        save_steps: Частота сохранения модели
        eval_steps: Частота оценки модели
        logging_steps: Частота логирования
        text_column: Название колонки с текстом
        label_column: Название колонки с метками
    """
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f"Используется устройство: {device}")
    
    if csv_path is None:
        csv_path = os.path.join(
            os.path.dirname(__file__),
            'main',
            'data_open_source_suicide_without_kaz.csv'
        )
    
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"Файл не найден: {csv_path}")
    
    logger.info(f"Инициализация токенизатора: {model_name}")
    tokenizer = XLMRobertaTokenizer.from_pretrained(model_name)
    
    logger.info("Загрузка и подготовка данных")
    train_texts, val_texts, test_texts, train_labels, val_labels, test_labels = load_and_prepare_data(
        csv_path, text_column=text_column, label_column=label_column
    )
    
    logger.info(f"Инициализация модели: {model_name}")
    model = XLMRobertaForSequenceClassification.from_pretrained(
        model_name,
        num_labels=2
    )
    model.to(device)
    
    # Создаем datasets
    train_dataset = SuicideTextDataset(train_texts, train_labels, tokenizer, max_length)
    val_dataset = SuicideTextDataset(val_texts, val_labels, tokenizer, max_length)
    test_dataset = SuicideTextDataset(test_texts, test_labels, tokenizer, max_length)
    
    # DataCollator для динамического padding
    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)
    
    training_args = TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=num_epochs,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        warmup_steps=warmup_steps,
        weight_decay=weight_decay,
        learning_rate=learning_rate,
        logging_dir=f"{output_dir}/logs",
        logging_steps=logging_steps,
        eval_strategy="steps",
        eval_steps=eval_steps,
        save_strategy="steps",
        save_steps=save_steps,
        load_best_model_at_end=True,
        metric_for_best_model="eval_f1",
        greater_is_better=True,
        save_total_limit=3,
        fp16=torch.cuda.is_available(),
        dataloader_num_workers=0,
        report_to="none"
    )
    
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        data_collator=data_collator,
        compute_metrics=compute_metrics,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=3)]
    )
    
    logger.info("Начало обучения")
    trainer.train()
    
    logger.info("Оценка на тестовом наборе")
    test_results = trainer.evaluate(test_dataset)
    logger.info(f"Результаты на тестовом наборе: {test_results}")
    
    logger.info(f"Сохранение модели в {output_dir}")
    trainer.save_model()
    tokenizer.save_pretrained(output_dir)
    
    logger.info("Обучение завершено")
    
    return trainer, test_results


if __name__ == "__main__":
    """
    При запуске `python train.py` сразу стартует обучение
    с преднастроенными параметрами без CLI аргументов.
    """
    train_model()


