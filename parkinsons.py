import pandas as pd
import requests
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score
import pickle

with open('parkinsons.data', 'rb') as file:
    data = pickle.load(file)
with open('parkinsons.names', 'rb') as file:
    names = pickle.load(file)

# Скачивание описания признаков
response = requests.get(file1)
names_content = response.text

# Разделение описания признаков на строки и извлечение имен признаков
feature_names = [line.split(':')[0].strip() for line in names_content.split('\n') if line.startswith('name')]

# Загрузка данных
data = pd.read_csv(file, names=feature_names)

# Разделение данных на признаки и метки
X = data.drop('status', axis=1)
y = data['status']

# Нормализация признаков
scaler = StandardScaler()
X = scaler.fit_transform(X)

# Разделение данных на обучающую и тестовую выборки
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Создание модели XGBoost
model = XGBClassifier()

# Обучение модели
model.fit(X_train, y_train)

# Предсказание на тестовой выборке
y_pred = model.predict(X_test)

# Вычисление точности модели
accuracy = accuracy_score(y_test, y_pred)
print(f'Accuracy: {accuracy*100.0}%')