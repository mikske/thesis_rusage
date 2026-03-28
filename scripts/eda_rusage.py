# %%
#проверяем, что есть в папке
from pathlib import Path
import pandas as pd
from IPython.core.display_functions import display

data_dir = Path('/Volumes/Extreme SSD/vkr_rusage')

print(data_dir.exists())
print(f"Содержимое папки:")
for p in data_dir.iterdir():
    print(" -", p.name)

# %%
#проверяем, что датасет существует
main_csv_name = "description.csv"
main_csv_path = data_dir / main_csv_name

print(f"Основной CSV: {main_csv_name}")
print(f"Существует: {main_csv_path.exists()}")

# %%
#загружаем датасет
if not main_csv_path.exists():
    raise FileNotFoundError(
        f"Файл {main_csv_path} не найден."
    )

cols = ["file_id", "book_name", "author", "age", "genre"]

df = pd.read_csv(
    main_csv_path,
    sep=None,  #определяем разделитель
    engine="python", #определяем разделитель
    header=None,  #указываем на отсусттвие строки-заголовка
    names=cols  #и указываем наименование колонок
    )

print(f"Форма датасета: {df.shape}")
print(f"Первые строки: ")
print(df.head(10))

print("\n Столбцы:")
print(df.columns.tolist())

#%%
#немного приведем данные впорядок для дальнейшей работы, просто чтобы быть уверенными
df["age"] = pd.to_numeric(df["age"], errors="coerce").astype("int64")

print(f"Уникальные значения 'age': ")
print(df["age"].value_counts(dropna=False).sort_index())

#%%
#визуальное представление исходного возрастного распределения
import matplotlib.pyplot as plt

age_counts_full = df["age"].value_counts().sort_index()

plt.figure(figsize=(6, 4))
age_counts_full.plot(kind="bar")
plt.title("Распределение возрастов в исходном корпусе")
plt.xlabel("Возраст")
plt.ylabel("Количество текстов")
plt.tight_layout()
plt.show()

print(age_counts_full)
#%%
#так как нас не интересуют тексты старшей школы (возрастные отметки: 16, 18), отфильтровываем их
import numpy as np

valid_ages = {3, 5, 6, 8, 11, 12}
df_filtered = df[df["age"].isin(valid_ages)].copy()

print(f"Размер df до фильтрации: {df.shape}")
print(f"Размер df_filtered (после фльтрации по возрасту): {df_filtered.shape}")
print(f"Распределение возрастов после фильтрации: ")
print(df_filtered["age"].value_counts().sort_index())

#делаем возрастной маппинг для удобства
age_to_group_label = {
    3: "0_младшая_группа",
    5: "1_дошкольная_группа",
    6: "1_дошкольная_группа",
    8: "2_младшая_школа",
    11: "2_младшая_школа",
    12: "2_младшая_школа"
}

df_filtered["age_group_label"] = df_filtered["age"].map(age_to_group_label)

#и сразу задаем числовое представление для удобства
group_label_to_id = {
    "0_младшая_группа": 0,
    "1_дошкольная_группа": 1,
    "2_младшая_школа": 2
}

df_filtered["age_group_id"] = df_filtered["age_group_label"].map(group_label_to_id).astype("Int64")

print(f"Распределение по возрастным группам (label):")
print(df_filtered["age_group_label"].value_counts().sort_index())

print(f"Распределение по возрастным группам (id):")
print(df_filtered["age_group_id"].value_counts().sort_index())
#%%
#визуальное представление возрастного распределения после
group_counts = df_filtered["age_group_id"].value_counts()

plt.figure(figsize=(6, 4))
group_counts.plot(kind="bar")
plt.title("Распределение укрупнённых возрастных групп")
plt.xlabel("Группа")
plt.ylabel("Количество текстов")
plt.tight_layout()
plt.show()

print(group_counts)
#%%
new_csv_path = "/Volumes/Extreme SSD/vkr_rusage/full_metadata.csv"
df_filtered.to_csv(new_csv_path, index=False)