import itertools

import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier, IsolationForest
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import f1_score, confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt


class Scoring:

    @staticmethod
    def map_columns(sample_data, data_description):
        field_mapping = data_description[['Field_in_data', 'Description_of_information']].dropna()
        renamed_columns = {}
        for field, description in zip(field_mapping['Field_in_data'], field_mapping['Description_of_information']):
            if description not in renamed_columns.values():
                renamed_columns[field] = description
            else:
                print(f"Попередження: пропускаємо дублювання для {description}.")
        sample_data.rename(columns=renamed_columns, inplace=True)
        sample_data = sample_data.loc[:, ~sample_data.columns.duplicated()]
        return sample_data

    @staticmethod
    def preprocess_data(data, target_column):
        clean_data = data.dropna(subset=[target_column])
        numeric_features = clean_data.select_dtypes(include=['number']).columns
        clean_data = clean_data[numeric_features]
        return clean_data

    @staticmethod
    def drop_unnecessary_columns(data: pd.DataFrame, columns_to_drop: list) -> pd.DataFrame:
        return data.drop(columns=columns_to_drop, errors='ignore')

    @staticmethod
    def perform_scoring(data, target_column):
        # Виділяємо характеристики та цільову колонку

        features = data.drop(columns=[target_column])
        target = data[target_column]

        # Поділ даних на навчальні та тестові
        X_train, X_test, y_train, y_test = train_test_split(features, target, test_size=0.2, random_state=42)

        # Заповнення пропущених значень
        X_train = X_train.fillna(X_train.mean())
        X_test = X_test.fillna(X_train.mean())

        # Перевірка залишкових NaN
        if X_train.isnull().sum().sum() > 0 or X_test.isnull().sum().sum() > 0:
            print("\nПропущені значення після заповнення:")
            X_train = X_train.dropna(axis=1)
            X_test = X_test.dropna(axis=1)

        # Навчання моделі
        model = GaussianNB()
        model.fit(X_train, y_train)

        predictions = model.predict(X_test)

        # Матриця помилок
        cm = confusion_matrix(y_test, predictions)
        print("Матриця помилок:")
        print(cm)

        Scoring.plot_confusion_matrix(y_test, predictions)

        print(f"F1-метрика моделі: {f1_score(y_test, model.predict(X_test)):.2f}")
        return model

    @staticmethod
    def analyze_feature_importance(data, target_column):
        # Виділення характеристик та цільової змінної
        features = data.drop(columns=[target_column])
        target = data[target_column]

        # Фільтрація числових колонок
        numeric_features = features.select_dtypes(include=['number'])
        numeric_features = numeric_features.fillna(numeric_features.mean())

        # Навчання моделі Random Forest
        model = RandomForestClassifier(random_state=47)
        model.fit(numeric_features, target)

        # Оцінка важливості характеристик
        feature_importances = model.feature_importances_
        importance_df = pd.DataFrame({
            'Feature': numeric_features.columns,
            'Importance': feature_importances
        }).sort_values(by='Importance', ascending=False)

        # Візуалізація
        plt.figure(figsize=(10, 6))
        plt.barh(importance_df['Feature'], importance_df['Importance'], color='skyblue')
        plt.xlabel('Важливість')
        plt.ylabel('Ознака')
        plt.title('Аналіз важливості характеристик')
        plt.gca().invert_yaxis()
        plt.show()

        return importance_df

    @staticmethod
    def plot_confusion_matrix(y_test, predictions):
        cm = confusion_matrix(y_test, predictions)

        # Побудова графіка
        disp = ConfusionMatrixDisplay(confusion_matrix=cm,
                                      display_labels=["Кредит повернено (0)", "Кредит не повернено (1)"])
        disp.plot(cmap=plt.cm.Blues)

        plt.title("Confusion Матриця")
        plt.show()

    @staticmethod
    def predict_loan(model, data):
        # new_data_df = pd.DataFrame([new_data])

        # Отримуємо список ознак, які використовувались при навчанні моделі
        numeric_features = model.feature_names_in_

        # Видаляємо зайві колонки, які не входять у список ознак моделі
        new_data_numeric = data[numeric_features]

        # Упевніться, що немає NaN значень
        if new_data_numeric.isnull().sum().sum() > 0:
            print("Попередження: у нових даних залишились пропущені значення.")
            new_data_numeric = new_data_numeric.fillna(0)

        # Прогноз
        prediction = model.predict(new_data_numeric)[0]

        print("Вхідні дані для прогнозу:")
        print(new_data_numeric)

        print(f"Результат передбачення: {'Кредит не повернено (1)' if prediction == 1 else 'Кредит повернено (0)'}")
        return prediction

    @staticmethod
    def detect_fraud(data):
        from sklearn.ensemble import IsolationForest
        from sklearn.preprocessing import StandardScaler

        # Вибір лише числових колонок
        numeric_data = data.select_dtypes(include=['number']).copy()

        # Перевірка на пропущені значення (NaN) та заповнення середніми
        numeric_data = numeric_data.fillna(numeric_data.mean())

        # Перевірка на некоректні значення до масштабування
        print("Перевірка на некоректні значення перед масштабуванням.")
        numeric_data.replace([np.inf, -np.inf], np.nan, inplace=True)
        if numeric_data.isnull().sum().sum() > 0:
            print("Заміна некоректних значень на 0.")
            numeric_data = numeric_data.fillna(0)

        # Масштабування даних
        scaler = StandardScaler()
        try:
            scaled_data = scaler.fit_transform(numeric_data)
        except ValueError as e:
            print(f"Помилка масштабування: {e}")
            return pd.DataFrame()

        # Перевірка на некоректні значення після масштабування
        print("Перевірка на некоректні значення після масштабування.")
        scaled_data = np.nan_to_num(scaled_data, nan=0.0, posinf=0.0, neginf=0.0)

        model = IsolationForest(contamination=0.05, random_state=42)
        try:
            model.fit(scaled_data)
        except ValueError as e:
            print(f"Помилка під час навчання IsolationForest: {e}")
            return pd.DataFrame()

        # Прогноз аномалій (аномалії = -1)
        predictions = model.predict(scaled_data)

        # Витягання індексів шахрайських записів
        fraudulent_indices = np.where(predictions == -1)[0]

        # Повернення шахрайських записів
        fraudulent_records = data.iloc[fraudulent_indices]

        print(f"Кількість шахрайських записів: {len(fraudulent_records)}")
        return fraudulent_records


if __name__ == "__main__":
    # Завантажимо sample_data - без description
    sample_data = pd.read_excel('sample_data.xlsx')
    data_description = pd.read_excel('data_description.xlsx')
    target_column_unmapped = 'loan_overdue'

    # Підготуємо дані
    preprocessed_data = Scoring.preprocess_data(sample_data, target_column_unmapped)
    columns_to_remove = ['Unnamed: 5', 'Application', 'user_id', 'created_at']
    cleaned_data = Scoring.drop_unnecessary_columns(preprocessed_data, columns_to_remove)

    # Перевіримо, чи є дані збалансованими:
    class_distribution = cleaned_data['loan_overdue'].value_counts()
    print(class_distribution)

    # Проведемо скорінг даних, які поки без description
    scoring_model_basic = Scoring.perform_scoring(cleaned_data, target_column_unmapped)
    Scoring.predict_loan(scoring_model_basic, cleaned_data)

    # Проведемо мапінг даних із description
    sample_mapped_data = Scoring.map_columns(sample_data, data_description)
    target_column_mapped = 'Позика протермінована'

    # Проведемо підготовку та скорінг даних які містять description
    preprocessed_mapped_data = Scoring.preprocess_data(sample_mapped_data, target_column_mapped)
    scoring_model = Scoring.perform_scoring(preprocessed_mapped_data, target_column_mapped)

    # Отримаємо рейтинг важливості кожної характеристики
    feature_importance = Scoring.analyze_feature_importance(preprocessed_mapped_data, target_column_mapped)
    print(feature_importance)

    # Приберемо не важливі характеристики
    columns_to_remove_using_rating = ['extension_amount', 'extension_days', 'Позика закрита', 'сума боргу',
                                      'product_overdue_start_day', 'нараховано стандартних відсотків',
                                      'Володіє нерухомістю', 'сума боргу', 'мінімальна сума по продукту']
    cleaned_preprocessed_data = Scoring.drop_unnecessary_columns(preprocessed_mapped_data,
                                                                 columns_to_remove_using_rating)

    # Знову проведемо підготовку та скорінг даних які очищені від менш важливих характеристик
    scoring_model_cleansed = Scoring.perform_scoring(cleaned_preprocessed_data, target_column_mapped)

    # Дамо прогноз чи поверне клієнт кредит
    Scoring.predict_loan(scoring_model_cleansed, cleaned_preprocessed_data)

    # Перевіримо наявність шахрайських записів
    fraudulent_records = Scoring.detect_fraud(cleaned_preprocessed_data)
