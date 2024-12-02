import pandas as pd
from lab3.DecisionSupport import DecisionSupport

multi_score = "total_multi_criteria_score"
brand = "brand"
model = "model"


class Olap:

    @staticmethod
    def olap_analysis(data, maximized_criteria, minimized_criteria):
        ranked_data = DecisionSupport.multi_criteria_evaluation(data, maximized_criteria, minimized_criteria)

        average_score = ranked_data[multi_score].mean()
        top_brand = ranked_data.iloc[0][[brand, multi_score]]
        top_model = ranked_data.iloc[0][[model, multi_score]]

        return {
            "average_score": average_score,
            "top_brand": top_brand,
            "top_model": top_model
        }


if __name__ == '__main__':
    data = pd.read_csv("..\lab3\laptops_with_weight.csv")
    maximized_criteria = ["ram", "ssd", "battery_life", "refresh_rate", "gpu_score", "user_rating", "release_year"]
    minimized_criteria = ["weight", "price"]
    laptop_efficiency = DecisionSupport.multi_criteria_evaluation(data, maximized_criteria, minimized_criteria)
    print(laptop_efficiency)

    olap_result = Olap.olap_analysis(data, maximized_criteria, minimized_criteria)
    print(f"Середній рейтинг: {olap_result['average_score']:.2f}\n")
    print(f"Найкращий бренд: {olap_result['top_brand']}\n")
    print(f"Найкраща модель: {olap_result['top_model']}\n")
