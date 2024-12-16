import pandas as pd
import numpy as np
from matplotlib import pyplot as plt

if __name__ == '__main__':

    # Дані про користувачів
    data = pd.DataFrame({
        "Користувач": ["Клієнт 1", "Клієнт 2", "Клієнт 3", "Клієнт 4"],
        "Щомісячний дохід": [5000, 7000, 3000, 10000],
        "Щомісячні витрати": [2000, 4000, 1500, 5000],
        "Кількість непогашених кредитів": [1, 0, 3, 1],
        "Кредитна історія (бал)": [700, 800, 500, 900],
    })

    # Нормалізація даних
    normalized_data = data.copy()
    for column in ["Щомісячний дохід", "Щомісячні витрати", "Кредитна історія (бал)"]:
        normalized_data[column] = data[column] / data[column].max()

    # Зважування критеріїв (рівні ваги)
    weights = np.array([0.4, 0.2, -0.3, 0.1])  # Витрати мають негативний вплив

    # Інтегрована оцінка
    normalized_data["Інтегрована оцінка"] = (
            normalized_data["Щомісячний дохід"] * weights[0] +
            (1 - normalized_data["Щомісячні витрати"]) * weights[1] +
            normalized_data["Кредитна історія (бал)"] * weights[2] +
            (1 - normalized_data["Кількість непогашених кредитів"] / 3) * weights[3]
    )

    # Сортування за ефективністю
    final_result = normalized_data.sort_values(by="Інтегрована оцінка", ascending=False)

    print("Результат оцінювання кредитоспроможності:")
    print(final_result[["Користувач", "Інтегрована оцінка"]])

    # Побудова графіка
    plt.figure(figsize=(10, 6))
    plt.bar(final_result["Користувач"], final_result["Інтегрована оцінка"], color="skyblue")
    plt.title("Інтегрована оцінка ефективності користувачів", fontsize=14)
    plt.xlabel("Користувач", fontsize=12)
    plt.ylabel("Інтегрована оцінка", fontsize=12)
    plt.grid(axis="y", linestyle="--", alpha=0.7)
    plt.tight_layout()
    plt.show()
