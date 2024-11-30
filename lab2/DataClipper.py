from lab1.AdditiveModel import AdditiveModel
import numpy as np
import matplotlib.pyplot as plt


def clip(data, lower_bound, upper_bound):
    return np.clip(data, lower_bound, upper_bound)


class DataClipper:

    @staticmethod
    def clip_stale_rate_and_show_plot(data, lower_bound, upper_bound):
        updated_data = np.clip(data, lower_bound, upper_bound)
        plt.figure(figsize=(10, 6))
        plt.plot(data, color='grey', linestyle='--', label="Оригінальні дані")
        plt.plot(updated_data, color='g', label="Обмежені дані", marker='o')
        plt.axhline(y=lower_bound, color='r', linestyle='--', label="Нижня межа")
        plt.axhline(y=upper_bound, color='r', linestyle='--', label="Верхня межа")
        plt.legend()
        plt.title(f"Обмеження оригінальних даних в межах [{lower_bound}, {upper_bound}]")
        plt.xlabel("Елемент")
        plt.ylabel("Значення")
        plt.show()

        return updated_data

    @staticmethod
    def clip_changing_rate_and_show_plot(data, lower_bound, upper_bound, change_rate):
        updated_lower_bounds = []
        for i in range(len(data)):
            new_value = lower_bound + i * change_rate
            updated_lower_bounds.append(new_value)

        updated_upper_bounds = []
        for i in range(len(data)):
            new_value = upper_bound + i * change_rate
            updated_upper_bounds.append(new_value)

        updated_data = []
        for value, lower, upper in zip(data, updated_lower_bounds, updated_upper_bounds):
            clipped_value = np.clip(value, lower, upper)
            updated_data.append(clipped_value)

        plt.figure(figsize=(10, 6))
        plt.plot(data, color='grey', linestyle='--', label="Оригінальні дані")
        plt.plot(updated_data, color='g', label="Обмежені дані", marker='o')
        plt.plot(updated_lower_bounds, color='r', linestyle='--', label="Адаптована Нижня межа")
        plt.plot(updated_upper_bounds, color='r', linestyle='--', label="Адаптована Верхня межа")
        plt.legend()
        plt.title(f"Обмеження оригінальних даних в межах [{lower_bound}, {upper_bound}]")
        plt.xlabel("Елемент")
        plt.ylabel("Значення")
        plt.show()

        return updated_data


if __name__ == '__main__':
    change_rate_stale = 0
    change_rate_rising = 20
    lower_bound = -100
    upper_bound = 100

    additive_model_stale = AdditiveModel(lowest_error_border=-150,
                                         highest_error_border=200,
                                         size=30,
                                         constant_value=0,
                                         change_rate=change_rate_stale)

    stale_data = additive_model_stale.generate_experimental_data()
    DataClipper.clip_stale_rate_and_show_plot(stale_data, lower_bound=lower_bound, upper_bound=upper_bound)

    additive_model_rising = AdditiveModel(lowest_error_border=-150,
                                          highest_error_border=200,
                                          size=30,
                                          constant_value=0,
                                          change_rate=change_rate_rising)

    changing_data = additive_model_rising.generate_experimental_data()
    DataClipper.clip_changing_rate_and_show_plot(changing_data, lower_bound=lower_bound, upper_bound=upper_bound,
                                                 change_rate=change_rate_rising)
