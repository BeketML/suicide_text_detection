from huggingface_hub import HfApi, create_repo, upload_folder
from pathlib import Path


api = HfApi()

# 1) Укажи свой ник и имя репозитория для модели
#    Пример: "nurbergen/xlm-roberta-suicide-detector"
model_repo_id = "BeketML/suicide-detection-text"

# 2) Создаём репозиторий (если ещё не создан)
create_repo(repo_id=model_repo_id, repo_type="model", exist_ok=True)

# Compute paths relative to this script file
here = Path(__file__).resolve().parent
model_dir = (here / ".." / ".." / "results").resolve()  # adjust depth if needed

print("Model dir:", model_dir)
print("Exists:", model_dir.is_dir())

api.upload_folder(
    folder_path=str(model_dir),
    repo_id=model_repo_id,
    repo_type="model",
)