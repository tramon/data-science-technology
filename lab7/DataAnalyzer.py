import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
from numpy.f2py.auxfuncs import throw_error

from lab2.PolynomialRegression import PolynomialRegression


class DataAnalyzer:

    @staticmethod
    def clean_excel_data(file_path):
        pd.set_option('future.no_silent_downcasting', True)
        data = pd.read_excel(file_path)

        data = data.replace({'n.a.': np.nan, 'not available': np.nan, 'not avilable': np.nan})
        data['SALES_BY_REGION'] = data['SALES_BY_REGION'].str.strip()

        # Перетворення числових колонок
        columns_to_convert = data.columns[2:]  # Колонки з продажами
        for col in columns_to_convert:
            # Видалення символів ',' та перетворення на числовий тип
            data[col] = data[col].replace(',', '', regex=True).astype(float, errors='ignore')

        # Визначення числових колонок (крім текстових)
        columns_to_convert = data.select_dtypes(include=[np.number]).columns

        # Очищення від'ємних значень (замінюємо на NaN)
        for col in columns_to_convert:
            data[col] = data[col].apply(lambda x: np.nan if x < 0 else x)

        cleaned_file_path = file_path.replace(".xlsx", "_Clean.csv")
        data.to_csv(cleaned_file_path, index=False)
        print(f"Очищений файл збережено як: {cleaned_file_path}")

        return data

    @staticmethod
    def analyze_sales_by_region(data):
        region_sales = data.groupby("SALES_BY_REGION").sum()
        return region_sales

    @staticmethod
    def plot_sales(data):
        monthly_sales = data.drop(columns=["SALES_ID", "SALES_BY_REGION"]).sum()
        monthly_sales.plot(kind='bar', figsize=(12, 6), color='blue',
                           title="Всього продажів по місяцям")
        plt.xlabel("Місяць")
        plt.ylabel("Продажі")
        plt.show()

    @staticmethod
    def plot_sales_by_region(cleaned_data):
        regions = cleaned_data['SALES_BY_REGION'].unique()
        months = cleaned_data.columns[2:]  # Вибираємо місяці як стовпці, починаючи з третього

        plt.figure(figsize=(14, 8))

        for region in regions:
            region_data = cleaned_data[cleaned_data['SALES_BY_REGION'] == region]
            if region_data.empty:
                continue

            # Підсумовуємо дані продажів для кожного місяця в регіоні
            monthly_sales = region_data.iloc[:, 2:].sum(axis=0).values.astype(float)

            plt.plot(months, monthly_sales, label=region, marker='o')  # Лінія продажів по регіону

        plt.title("Продажі по місяцях з деталізацією по регіонах")
        plt.xlabel("Місяці")
        plt.ylabel("Обсяг продажів")
        plt.legend(title="Регіони")
        plt.grid(True)
        plt.xticks(rotation=45)  # Розвертаємо підписи місяців для зручності
        plt.tight_layout()  # Оптимізує розташування графіка
        plt.show()

    @staticmethod
    def fit_polynomial_model(region_data, degree):
        x_axis = np.arange(1, len(region_data) + 1)
        y_axis = region_data.values

        polynomial_model = PolynomialRegression(degree)
        polynomial_model.learn_using_lsm(x_axis, y_axis)

        return polynomial_model

    @staticmethod
    def predict_sales_for_next_months(model, current_months, future_months=6):
        x_future = np.arange(current_months + 1, current_months + future_months + 1)
        predictions = model.predict(x_future)
        return x_future, predictions

    @staticmethod
    def plot_sales_prediction(region, months, sales, future_months, predictions):
        plt.figure(figsize=(12, 6))
        plt.plot(months, sales, marker='o', label='Історичні дані', color='blue')
        plt.plot(future_months, predictions, marker='x', label='Прогноз', color='green', linestyle='--')
        plt.axvline(x=months[-1], color='red', linestyle='--', label='Поточний місяць')
        plt.title(f"Прогноз продажів для регіону {region}")
        plt.xlabel("Місяці")
        plt.ylabel("Продажі")
        plt.legend()
        plt.grid(True)
        plt.show()

    @staticmethod
    def visualize_sales_and_predictions(cleaned_data):
        regions = cleaned_data['SALES_BY_REGION'].unique()
        plt.figure(figsize=(12, 8))

        for region in regions:
            region_data = cleaned_data[cleaned_data['SALES_BY_REGION'] == region]

            if region_data.empty:
                print(f"Для регіону {region} немає даних.")
                continue

            # Місяці та продажі
            months = np.arange(1, 13)  # 12 місяців
            sales = region_data.iloc[:, 2:].sum(axis=0).values.astype(float)  # Дані продажів для регіону

            # Перевірка наявності коректних даних
            if np.isnan(sales).any() or len(sales) != 12:
                print(f"Дані для регіону {region} некоректні. Пропускаємо.")
                continue

            # Навчання моделі
            degree = 2
            poly_model = PolynomialRegression(degree=degree)
            poly_model.learn_using_lsm(months, sales)

            # Прогнозування
            future_months, predictions = DataAnalyzer.predict_sales_for_next_months(poly_model, len(months))

            # Об'єднуємо історію та прогноз
            all_months = np.concatenate((months, future_months))
            all_sales = np.concatenate((sales, predictions))

            # Графік
            plt.plot(all_months, all_sales, label=region)
            plt.scatter(months, sales, s=10)  # Історичні дані
            plt.axvline(x=12.5, color='red', linestyle='--', label='Поточний місяць' if region == regions[0] else "")

        plt.title("Фактичні дані та прогноз продажів по всіх ринках")
        plt.xlabel("Місяці")
        plt.ylabel("Продажі")
        plt.legend()
        plt.grid(True)
        plt.show()

    @staticmethod
    def extrapolate_by_region(cleaned_data, region):
        regions = cleaned_data['SALES_BY_REGION'].dropna().unique()  # Унікальні регіони, виключаючи nan
        for r in regions:
            if r == region:
                region_data = cleaned_data[cleaned_data['SALES_BY_REGION'] == r]

                # Місяці та продажі
                months = np.arange(1, 13)  # 12 місяців
                sales = region_data.iloc[:, 2:].sum(axis=0).values.astype(float)  # Дані продажів для регіону

                # Навчання моделі
                poly_model = PolynomialRegression(degree=2)
                poly_model.learn_using_lsm(months, sales)

                # Прогнозування та Візуалізація окремо по кожному ринку факт + прогноз
                future_months, predictions = DataAnalyzer.predict_sales_for_next_months(poly_model, len(months))
                DataAnalyzer.plot_sales_prediction(r, months, sales, future_months, predictions)
            else:
                throw_error("There is no such region")


if __name__ == '__main__':
    analyzer = DataAnalyzer()
    source_file_path = "Data_Set_6.xlsx"

    data = analyzer.clean_excel_data(source_file_path)
    sales_by_region = analyzer.analyze_sales_by_region(data)
    print(sales_by_region)

    cleaned_file_path = 'Data_Set_6_Clean.csv'
    cleaned_data = pd.read_csv(cleaned_file_path)

    analyzer.plot_sales(data)
    analyzer.plot_sales_by_region(data)

    analyzer.extrapolate_by_region(cleaned_data, "UAQ")
    DataAnalyzer.visualize_sales_and_predictions(cleaned_data)
