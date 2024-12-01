import pandas as pd
import matplotlib.pyplot as plt

score = "total_score"
multi_score = "total_multi_criteria_score"


class DecisionSupport:

    @staticmethod
    def normalize_data(data, max_criteria, min_criteria):
        normalized_data = data.copy()
        for criterion in max_criteria:
            normalized_data[criterion] = (data[criterion] - data[criterion].min()) / (
                    data[criterion].max() - data[criterion].min())

        for criterion in min_criteria:
            normalized_data[criterion] = (data[criterion].max() - data[criterion]) / (
                    data[criterion].max() - data[criterion].min())
            normalized_data[score] = normalized_data[max_criteria + min_criteria].sum(axis=1)
            sorted = normalized_data.sort_values(by=score, ascending=False)
        return sorted

    @staticmethod
    def multi_criteria_evaluation(data, maximized_criteria, minimized_criteria):
        """
        Розраховує багато-критеріальний рейтинг, враховуючи лише числові значення.
        """
        weights = data.iloc[-1, :].drop(["brand", "model"], errors="ignore")

        # Перетворення ваг у числовий формат; нечислові значення виключаються
        weights = pd.to_numeric(weights, errors="coerce").dropna().to_dict()
        data = data.iloc[:-1].copy()

        numeric_criteria = maximized_criteria + minimized_criteria
        numeric_criteria = [criterion for criterion in numeric_criteria if criterion in weights]

        for criterion in numeric_criteria:
            if criterion in maximized_criteria:
                data[criterion] = pd.to_numeric(data[criterion], errors="coerce")
                data[criterion] = (data[criterion] - data[criterion].min()) / (
                        data[criterion].max() - data[criterion].min())
            elif criterion in minimized_criteria:
                data[criterion] = pd.to_numeric(data[criterion], errors="coerce")
                data[criterion] = (data[criterion].max() - data[criterion]) / (
                        data[criterion].max() - data[criterion].min())

        # Виключення нечислових значень
        data = data.dropna(subset=numeric_criteria)

        # Застосування ваг до нормалізованих даних
        for criterion, weight in weights.items():
            if criterion in numeric_criteria:
                data[criterion] *= weight

        data["total_multi_criteria_score"] = data[numeric_criteria].sum(axis=1)

        return data.sort_values(by="total_multi_criteria_score", ascending=False)

    @staticmethod
    def plot_rating(data, characteristics, criteria):
        plt.figure(figsize=(12, 6))
        plt.bar(data[characteristics], data[criteria], color="blue")
        plt.title(f"Рейтинг товарів по {criteria}")
        plt.xlabel("Товар")
        plt.ylabel("Загальний рейтинг")
        plt.xticks(rotation=45)
        plt.grid(axis="y", linestyle="--", alpha=0.7)
        plt.show()


if __name__ == '__main__':
    data = pd.read_csv("laptops.csv")

    characteristics_to_show = "model"
    alternate_characteristics_to_show = "brand"

    maximized_criteria = ["ram", "ssd", "battery_life", "refresh_rate", "gpu_score", "user_rating", "release_year"]
    minimized_criteria = ["weight", "price"]

    rated_data = DecisionSupport.normalize_data(data, maximized_criteria, minimized_criteria)

    print("Рейтинг товарів за одним критерієм:")
    print(rated_data[[characteristics_to_show, score]].to_string(formatters={score: "{:.2f}".format}))
    DecisionSupport.plot_rating(rated_data, characteristics_to_show, score)

    data_with_weight = pd.read_csv("laptops_with_weight.csv")
    data_with_weight_rated = DecisionSupport.multi_criteria_evaluation(data_with_weight,
                                                                       maximized_criteria,
                                                                       minimized_criteria)

    print("\nБагато-критеріальний Рейтинг товарів, що враховує вагу кожного фактора:")
    print(data_with_weight_rated[[characteristics_to_show, multi_score]]
          .to_string(formatters={multi_score: "{:.2f}".format}))
    DecisionSupport.plot_rating(data_with_weight_rated, characteristics_to_show, multi_score)
