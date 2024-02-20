import pandas as pd
import urllib.request

# URL второго датасета
names_url = "https://storage.yandexcloud.net/academy.ai/practica/parkinsons.names"

# Загрузка файла с информацией о признаках
with urllib.request.urlopen(names_url) as response:
    names_info = response.read().decode('utf-8')

print(names_info)

# Загрузка данных
url = "https://storage.yandexcloud.net/academy.ai/practica/parkinsons.data"
data = pd.read_csv(url)

# Вывод первых нескольких строк данных для ознакомления
print(data.head())

# Проверка наличия пропущенных значений
print(data.isnull().sum())

# Вывод информации о типах данных и статистики
print(data.info())
print(data.describe())

from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

# Разделение данных на признаки (X) и метки (y)
X = data.drop(['name', 'status'], axis=1)
y = data['status']

# Нормализация признаков
scaler = StandardScaler()
X_normalized = scaler.fit_transform(X)

# Разделение данных на обучающую и тестовую выборки (80% - обучающая, 20% - тестовая)
X_train, X_test, y_train, y_test = train_test_split(X_normalized, y, test_size=0.2, random_state=42)

import xgboost as xgb

# Создание и обучение модели XGBoost
model = xgb.XGBClassifier()
model.fit(X_train, y_train)

from sklearn.metrics import accuracy_score

# Предсказание меток для тестовой выборки
y_pred = model.predict(X_test)

# Оценка точности модели
accuracy = accuracy_score(y_test, y_pred)
print("Точность модели на тестовой выборке:", accuracy)
