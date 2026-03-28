#%%
import os
import shutil
import pandas as pd

BASE_DIR = "/Volumes/Extreme SSD/vkr_rusage"

METADATA_CSV = os.path.join(BASE_DIR, "full_metadata.csv")
PREVIEWS_DIR = os.path.join(BASE_DIR, "previews")
SELECTED_DIR = os.path.join(BASE_DIR, "selected_previews")

FILE_COLUMN = "file_id"
ADULT_AGES = [16, 18]
#%%
#df = pd.read_csv(METADATA_CSV)
#print(df.head())
#%%
def main():
    print("Читаю метаданные из:", METADATA_CSV)
    df = pd.read_csv(METADATA_CSV)

    if FILE_COLUMN not in df.columns:
        raise ValueError(
            f"В csv нет колонки '{FILE_COLUMN}'. "
            f"Доступные колонки: {list(df.columns)}"
        )

    if "age" not in df.columns:
        raise ValueError("В csv нет колонки 'age', не из чего фильтровать взрослых.")

    #приводим age к числу
    df["age"] = pd.to_numeric(df["age"], errors="coerce")

    #убираем взрослых (16 и 18)
    df_filtered = df[~df["age"].isin(ADULT_AGES)].copy()

    print("Всего строк в full_metadata:", len(df))
    print("После фильтра по возрасту (без 16 и 18):", len(df_filtered))
    print("Распределение age в df_filtered:")
    print(df_filtered["age"].value_counts().sort_index())

    #уникальные file_id
    file_names = df_filtered[FILE_COLUMN].astype(str).unique()
    print("Уникальных file_id в отфильтрованных данных:", len(file_names))

    # ЧИСТИМ и пересоздаём selected_previews
    if os.path.exists(SELECTED_DIR):
        print("Удаляю старую папку:", SELECTED_DIR)
        shutil.rmtree(SELECTED_DIR)
    os.makedirs(SELECTED_DIR, exist_ok=True)
    print("Создана пустая папка:", SELECTED_DIR)

    copied = 0
    missing = []
    skipped_artifacts = 0

    #копируем файлы
    for name in file_names:
        #на всякий случай пропускаем ._ имена
        if name.startswith("._"):
            skipped_artifacts += 1
            continue

        src = os.path.join(PREVIEWS_DIR, name)

        if not os.path.isfile(src):
            missing.append(name)
            continue

        #ещё одна проверка на ._
        if os.path.basename(src).startswith("._"):
            skipped_artifacts += 1
            continue

        dst = os.path.join(SELECTED_DIR, name)
        shutil.copy2(src, dst)
        copied += 1

    #удаляем ._ внутри целевой папки
    removed_artifacts = 0
    for fname in os.listdir(SELECTED_DIR):
        if fname.startswith("._"):
            os.remove(os.path.join(SELECTED_DIR, fname))
            removed_artifacts += 1

    print(f"Скопировано файлов: {copied}")
    print(f"Файлов из метаданных не найдено в previews: {len(missing)}")
    if missing:
        print("Примеры отсутствующих:", missing[:10])
    print(f"Пропущено ._ артефактов (на этапе копирования): {skipped_artifacts}")
    print(f"Удалено ._ артефактов в {SELECTED_DIR}: {removed_artifacts}")

#%%
if __name__ == "__main__":
    main()
#%%
import pandas as pd

META = "/Volumes/Extreme SSD/vkr_rusage/full_metadata.csv"
df = pd.read_csv(META)

# убедимся, что age_group_id есть
print(df.columns)
df_id2 = df[df["age_group_id"] == 2]

print("Всего файлов для age_group_id 2:", len(df_id2))
df_id2["file_id"].head(20).tolist()
sorted(df_id2["file_id"].astype(str).unique())[:30]

