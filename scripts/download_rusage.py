from pathlib import Path
import sys

from kaggle.api.kaggle_api_extended import KaggleApi

DATASET = "oldaandozerskaya/fiction-corpus-for-agebased-text-classification"
TARGET_DIR = Path('/Volumes/Extreme SSD/vkr_rusage')

def main():
    try:
        import kaggle
    except ImportError:
        print(
            "Модуль 'kaggle' не установлен.\n"
            "Активируй окружение vkr_rusage и установи его командой:\n"
            "    pip install kaggle\n"
        )
        sys.exit(1)

    api = KaggleApi()
    try:
        api.authenticate()
    except Exception as e:
        print(
            "Не удалось аутентифицироваться в Kaggle API.\n"
            "Убедись, что в конфигурации запуска PyCharm задана переменная KAGGLE_API_TOKEN\n"
            "с твоим токеном (KGAT_...).\n"
            f"Техническая ошибка: {e}"
        )
        sys.exit(1)

    TARGET_DIR.mkdir(parents=True, exist_ok=True)
    print(f"Папка для данных: {TARGET_DIR}")
    print(f"Содержимое для скачивания: {[p.name for p in TARGET_DIR.iterdir()] or 'пусто'})")

    #скачиваем датасет
    print(f"Скачиваем датасет: {DATASET} в {TARGET_DIR}")
    api.dataset_download_files(
        DATASET,
        path=str(TARGET_DIR),
        unzip=True,
    )

    print(f"Содержимое папки:")
    for p in TARGET_DIR.iterdir():
        print(" -", p.name)

if __name__ == "__main__":
    main()