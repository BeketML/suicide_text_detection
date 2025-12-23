1. Collect open source related to suicide all links to data in all_links.txt
2. Collect all data related with suicide 
Конечно, вот пошаговый конспект того, что мы сделали, начиная с установки WSL, чтобы PyTorch работал с твоим GPU в Windows через WSL:

---

## 1. Установка WSL и Ubuntu

1. Проверили наличие WSL: `wsl --version`
2. Установили дистрибутив Ubuntu 22.04:

   ```powershell
   wsl --install -d Ubuntu-22.04
   ```
3. Создали UNIX-пользователя для Ubuntu (username: `beket`).
4. Проверили доступ к WSL:

   ```bash
   wsl -d Ubuntu-22.04
   ```

---

## 2. Удаление лишних дистрибутивов (опционально)

1. Список всех дистрибутивов:

   ```powershell
   wsl --list --verbose
   ```
2. Оставили только нужный дистрибутив, удалили остальные:

   ```powershell
   wsl --unregister <distro_name>
   ```

---

## 3. Настройка CUDA в WSL

1. Установили CUDA Toolkit 12.9:

   ```bash
   sudo apt install cuda-toolkit-12-9 -y
   ```
2. Проверили версию компилятора:

   ```bash
   nvcc --version
   ```
3. Если `nvcc` не найден, установили пакет:

   ```bash
   sudo apt install nvidia-cuda-toolkit
   ```
4. Добавили CUDA в PATH:

   ```bash
   echo 'export PATH=/usr/local/cuda-12.9/bin:$PATH' >> ~/.bashrc
   echo 'export LD_LIBRARY_PATH=/usr/local/cuda-12.9/lib64:$LD_LIBRARY_PATH' >> ~/.bashrc
   source ~/.bashrc
   ```
5. Проверили: `nvcc --version` → CUDA 12.9

---

## 4. Создание виртуального окружения в WSL

1. Установили пакет для создания виртуальных окружений:

   ```bash
   sudo apt install python3.10-venv -y
   ```
2. Создали виртуальное окружение:

   ```bash
   python3 -m venv venv_wsl
   source venv_wsl/bin/activate
   ```

---

## 5. Установка PyTorch, torchvision и torchaudio под CUDA 12.9

1. Использовали правильные версии для совместимости с драйвером (576.88 поддерживает CUDA 12.9):

   ```bash
   pip install --upgrade pip
   pip install torch==2.9.1+cu129 torchvision==0.24.1+cu129 torchaudio==2.9.1+cu129 --index-url https://download.pytorch.org/whl/cu129
   ```
2. Проверили работу GPU в PyTorch:

   ```python
   import torch
   print(torch.version.cuda)       # 12.9
   print(torch.cuda.is_available()) # True
   print(torch.cuda.get_device_name())
   ```

---

## 6. Результат

* WSL Ubuntu видит GPU RTX 5090 через драйвер Windows.
* PyTorch работает с CUDA 12.9.
* Все зависимости PyTorch корректно установлены.

---

Если хочешь, я могу сделать **короткий список команд, которые нужно выполнять теперь для работы с PyTorch в WSL**, чтобы больше не запутаться.

Хочешь, чтобы я это сделал?
