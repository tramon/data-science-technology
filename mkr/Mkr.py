import pandas as pd
from matplotlib import pyplot as plt

integrated_score = 'integrated_score'
model = 'model'
brand = 'brand'


class Mkr:

    @staticmethod
    def calculate_integrated_score(data):
        # Нормалізація ціни (чим менше, тим краще)
        data['price_norm'] = data['price'].min() / data['price']

        # Нормалізація продуктивності
        performance_features = ['ram', 'ssd', 'gpu_score', 'refresh_rate']
        for feature in performance_features:
            data[f'{feature}_norm'] = data[feature] / data[feature].max()
        data['performance_norm'] = data[[f'{feature}_norm' for feature in performance_features]].mean(axis=1)

        # Нормалізація тривалості роботи батареї (battery life)
        data['battery_norm'] = data['battery_life'] / data['battery_life'].max()

        # Інтегрована оцінка
        data['integrated_score'] = data[['price_norm', 'performance_norm', 'battery_norm']].mean(axis=1)
        return laptops.sort_values(by='integrated_score', ascending=False).head(10)

    @staticmethod
    def plot(data, attribute_to_show):
        plt.bar(data[attribute_to_show], data[integrated_score])
        plt.title("Інтегрована оцінка ноутбуків")
        plt.xlabel("Назва товару")
        plt.ylabel("Оцінка")
        plt.show()


if __name__ == '__main__':
    laptops = pd.read_csv("laptops.csv")

    top_10_laptops = Mkr.calculate_integrated_score(laptops)
    Mkr.plot(top_10_laptops, model)
