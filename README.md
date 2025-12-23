## Suicide Text Detection with XLM-RoBERTa

This repository contains the full workflow for training, evaluating, and serving a **suicide risk text classification model** based on **[FacebookAI/xlm-roberta-base](https://huggingface.co/FacebookAI/xlm-roberta-base)**.  
The model is trained on a large collection of multilingual (primarily Russian and English) social media posts and predicts whether a given text expresses **suicidal ideation** (`suicide`) or **non‑suicidal** content (`non_suicide`).

The project includes:

- **Data preparation & cleaning** (merging multiple open-source datasets, language detection, normalization).
- **Exploratory Data Analysis (EDA)**.
- **Model training** using Hugging Face Transformers (XLM‑RoBERTa).
- **Saved model artifacts** in `results/` (including `model.safetensors` and tokenizer).
- **Inference API** using FastAPI.
- **Streamlit web interface** for interactive testing.
- **Utilities** for uploading the trained model and dataset to Hugging Face Hub.
<img width="843" height="944" alt="streamlit_ui" src="https://github.com/user-attachments/assets/0b97af5a-51e3-4066-8be4-c1cd1bc6d4ca" />

---

## Repository Structure

```text
.
├── README.md                     # This file
├── results/                      # Trained model weights & checkpoints
│   ├── config.json               # Hugging Face model config (model_type: xlm-roberta)
│   ├── model.safetensors         # Fine-tuned model weights
│   ├── tokenizer_config.json     # Tokenizer configuration
│   ├── sentencepiece.bpe.model   # SentencePiece model for XLM-R
│   ├── training_args.bin         # Saved TrainingArguments from HF Trainer
│   ├── special_tokens_map.json   # Special tokens for tokenizer
│   ├── checkpoint-7000/          # Intermediate HF checkpoints
│   ├── checkpoint-8000/
│   └── checkpoint-8500/
└── suicide_final_project/
    ├── requirements.txt          # Python dependencies for the project
    ├── results.png               # Example evaluation/visualization (e.g., metrics plot)
    └── final/
        ├── all_data_cleaned.csv  # Preprocessed training data (text + label + metadata)
        ├── final_pre_process.ipynb  # Data collection & preprocessing notebook
        ├── final_eda.ipynb          # Exploratory Data Analysis (EDA) notebook
        ├── train.py                 # XLM-RoBERTa fine-tuning script
        ├── inference.py             # FastAPI inference service
        ├── streamlit_app.py         # Streamlit front-end for manual testing
        └── helpers/
            └── kaznlp/              # KazNLP toolkit (tokenization, morphology, language ID)
                ├── lid/             # Language identification models (char/word NB, etc.)
                ├── morphology/      # Morphological analyzer & tagger
                ├── tokenization/    # Tokenizers for Kazakh text
                └── ...
```

> **Note:** Raw source datasets (original CSV/JSON from external providers) are **not** included here.  
> The merged and cleaned training data is available as `suicide_final_project/final/all_data_cleaned.csv`.

---

## Data Pipeline

The data pipeline is implemented in the Jupyter notebooks under `suicide_final_project/final/`:

### 1. `final_pre_process.ipynb`

- Loads multiple open-source suicide‑related datasets (Reddit, Twitter, Kaggle, Mendeley, etc.).
- Standardizes schema to `text` + `label` (+ `source`, `source_link`, `language`).
- Applies:
  - Text cleaning (lowercasing, punctuation removal, emoji handling, de‑duplication, NaN removal).
  - **Language detection** using the [kaznlp](https://github.com/makazhan/kaznlp) toolkit:
    - Detects Russian vs. Kazakh vs. other languages using Naive Bayes models (`lid/lidnb.py`).
  - Exports a final merged dataset to:

    ```text
    suicide_final_project/final/all_data_cleaned.csv
    ```

### 2. `final_eda.ipynb`

- **Dataset overview**: size, column types, label distribution.
- **Length analysis**: character and token length distributions, per‑label statistics.
- **Token statistics**:
  - Most common tokens for Russian and English.
  - Separate token distributions for `suicide` vs `non_suicide`.
  - Top Russian and English bigrams/trigrams for suicidal content.
- **Visualizations**:
  - Histograms of text lengths.
  - Word clouds for suicidal texts (Russian / English).
- **Source & language analysis**:
  - Distribution of posts across data sources (`source`).
  - Label distribution per source.
  - Language vs label vs source interactions.

---

## Model Training

The main training script is `suicide_final_project/final/train.py`.

### Model

- Base model: `FacebookAI/xlm-roberta-base`
- Task: **binary text classification**
  - `label = 0` → `non_suicide`
  - `label = 1` → `suicide`
- Uses `transformers.XLMRobertaForSequenceClassification`
- Input is taken from the `clean_text` column of `all_data_cleaned.csv`

### Training script (`train.py`)

Key components:

- `SuicideTextDataset`: custom `torch.utils.data.Dataset` that:
  - Takes lists of texts and labels.
  - Uses `XLMRobertaTokenizer` for tokenization (max length configurable, default `512`).
- `load_and_prepare_data`:
  - Reads `all_data_cleaned.csv`.
  - Keeps `clean_text` and `label` columns.
  - Splits data into **train / validation / test** using `train_test_split` with stratification.
- `compute_metrics`:
  - Computes **accuracy**, **weighted F1**, **precision**, and **recall** using `sklearn`.
- `train_model`:
  - Configures `TrainingArguments` (epochs, batch size, learning rate, weight decay, logging & saving steps, etc.).
  - Uses `transformers.Trainer` with:
    - Dynamic padding via `DataCollatorWithPadding`.
    - `EarlyStoppingCallback` for regularization.
  - Saves the best model and tokenizer into the `results/` directory.

### Model Performance

The final fine‑tuned model on the held‑out test set achieves:

- **Accuracy**: 0.93  
- **F1‑score**: 0.93  
- **Precision**: 0.93  
- **Recall**: 0.93  
- **Eval loss**: 0.19  

Metrics are computed with weighted averaging across the two classes (`non_suicide`, `suicide`).
<img width="1411" height="599" alt="results" src="https://github.com/user-attachments/assets/972ec14c-ddcf-42b7-a830-9d4e68d4d59b" />

#### Example: Run training

From the `suicide_final_project/final` directory:

```bash
cd suicide_final_project/final
python train.py
```

By default, the script expects:

- Data: `../data/open_source_suicide_without_kaz.csv` or `all_data_cleaned.csv` (you can adapt the path in the script).
- Output: model and tokenizer saved to `./results` (which is symlinked / copied to the top-level `results/` folder).

---

## Inference API (FastAPI)

The file `suicide_final_project/final/inference.py` exposes the trained model as a **REST API** using **FastAPI**.

### Features

- Loads the fine‑tuned XLM‑RoBERTa model from the `results/` directory.
- Automatically selects **GPU** if available (`torch.cuda.is_available()`), logs which device is used.
- Provides:
  - `GET /health` — health check endpoint.
  - `POST /predict` — main inference endpoint.
- Returns:
  - `label_id`: integer class index (`0` or `1`).
  - `label`: human‑readable label (`non_suicide` / `suicide`).
  - `confidence`: probability of the predicted class.
  - `device`: `"cuda"` or `"cpu"`.

### Run the API (WSL / local)

```bash
cd suicide_final_project/final

# (Optional) create & activate a virtual environment
python -m venv venv
source venv/bin/activate  # On Windows PowerShell: .\\venv\\Scripts\\Activate

# Install dependencies
pip install -r ../requirements.txt

# Start the API server
uvicorn inference:app --host 0.0.0.0 --port 8000 --reload
```

Then open in browser:

- Swagger UI: `http://localhost:8000/docs`
- Health check: `http://localhost:8000/health`

Example request:

```bash
curl -X POST "http://localhost:8000/predict" \
  -H "Content-Type: application/json" \
  -d '{"text": "Мне очень плохо, не знаю как жить дальше"}'
```

---

## Streamlit Web Interface

The file `suicide_final_project/final/streamlit_app.py` provides a simple UI for manual testing.

- Sends requests to the FastAPI `/predict` endpoint.
- Displays:
  - predicted label,
  - confidence score,
  - device used,
  - raw JSON response.

### Run Streamlit app

1. Убедись, что FastAPI сервер запущен (см. раздел выше).
2. В другом терминале:

```bash
cd suicide_final_project/final
streamlit run streamlit_app.py
```

3. Открой браузер по адресу, который выведет Streamlit (обычно `http://localhost:8501`).

По умолчанию фронт стучится в `http://localhost:8000/predict`.  
Если ты используешь публичный URL (например, через ngrok), можно передать его через переменную окружения:

```bash
export SUICIDE_API_URL="https://your-ngrok-id.ngrok-free.app/predict"
streamlit run streamlit_app.py
```

---

## Using the Trained Model Directly

Ты можешь использовать модель и токенайзер напрямую из Python без API.

```python
from pathlib import Path
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification

model_dir = Path("results")  # папка с config.json и model.safetensors

tokenizer = AutoModelForSequenceClassification.from_pretrained(str(model_dir))
model = AutoModelForSequenceClassification.from_pretrained(str(model_dir))
model.eval()

ID2LABEL = {0: "non_suicide", 1: "suicide"}

def predict(text: str):
    inputs = tokenizer(
        text,
        return_tensors="pt",
        truncation=True,
        padding=True,
        max_length=512,
    )
    with torch.no_grad():
        outputs = model(**inputs)
        probs = torch.softmax(outputs.logits, dim=-1)
    pred_id = int(torch.argmax(probs, dim=-1).item())
    return {
        "label_id": pred_id,
        "label": ID2LABEL[pred_id],
        "confidence": float(probs[0, pred_id].item()),
    }

print(predict("Мне грустно и тяжело жить"))
```

Если ты уже загрузил модель на Hugging Face Hub, можно заменить `model_dir` на id репозитория:

```python
from transformers import AutoTokenizer, AutoModelForSequenceClassification

model_id = "your-username/xlm-roberta-suicide-detector"
tokenizer = AutoTokenizer.from_pretrained(model_id)
model = AutoModelForSequenceClassification.from_pretrained(model_id)
```

---

## Upload to Hugging Face Hub

Скрипт `suicide_final_project/final/upload_huggingface.py` демонстрирует, как:

- Создать репозиторий модели (`repo_type="model"`) и загрузить туда содержимое `results/`.
- Создать репозиторий датасета (`repo_type="dataset"`) и загрузить `all_data_cleaned.csv`.

Пример (фрагмент кода):

```python
from pathlib import Path
from huggingface_hub import HfApi, create_repo, upload_folder, upload_file

api = HfApi()

# 1. Create model repo (or reuse if it exists)
model_repo_id = "your-username/xlm-roberta-suicide-detector"
create_repo(repo_id=model_repo_id, repo_type="model", exist_ok=True)
api.upload_folder(
    folder_path="results",
    repo_id=model_repo_id,
    repo_type="model",
)

# 2. Create dataset repo and upload processed CSV
dataset_repo_id = "your-username/suicide-text-dataset"
create_repo(repo_id=dataset_repo_id, repo_type="dataset", exist_ok=True)
api.upload_file(
    path_or_fileobj="suicide_final_project/final/all_data_cleaned.csv",
    path_in_repo="data/all_data_cleaned.csv",
    repo_id=dataset_repo_id,
    repo_type="dataset",
)
```

---

## Environment Setup

### Requirements

- Python 3.10+
- (Optional) GPU with CUDA support for faster training/inference.

Установи зависимости:

```bash
cd suicide_final_project
pip install -r requirements.txt
```

Основные пакеты:

- `transformers`
- `torch`
- `pandas`, `numpy`
- `scikit-learn`
- `matplotlib`, `seaborn`, `polars`
- `fastapi`, `uvicorn`, `streamlit`
- `huggingface_hub`, `sentencepiece`
- `requests`, `bs4`, `playwright`, `selenium`, `webdriver-manager`, `jmespath`, `python-dotenv`

---

## Notes & Next Steps

- **Safety & Ethics**: This model deals with highly sensitive content (suicide and mental health).  
  It is intended for research and educational purposes and **must not** be used as a replacement for professional mental health care, crisis intervention, or medical diagnosis.
- Consider adding:
  - Bias and fairness analysis for different languages/demographics.
  - More robust evaluation (per‑dataset metrics, calibration, error analysis).
  - Integration tests for the API and Streamlit app.
  - CI/CD (e.g., GitHub Actions) for linting, testing, and deployment.
- You can also deploy the model as:
  - A **Hugging Face Space** (with Streamlit or Gradio).
  - A containerized FastAPI app on services like Render, Railway, Fly.io, or any cloud provider.

---

If you have any questions or want to extend this project (e.g., multi‑label classification, ranking severity levels, or adding more languages), feel free to open an issue or PR in this repository.


