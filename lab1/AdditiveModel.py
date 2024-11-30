import numpy as np
import matplotlib.pyplot as plt


# Адитивна Модель зміни досліджуваного процесу
class AdditiveModel:
    def __init__(self, lowest_error_border, highest_error_border, size, constant_value, change_rate):
        """
        lowest_error_border - Нижня межа рівномірного розподілу похибки.
        highest_error_border - Верхня межа рівномірного розподілу похибки.
        size - Кількість точок для генерації.
        constant_value - Постійне значення процесу.
        change_rate - Швидкість зміни процесу (якщо позитивне - зростає, якщо негативне - спадає).
        """
        self.lowest_error_border = lowest_error_border
        self.highest_error_border = highest_error_border
        self.size = size
        self.constant_value = constant_value
        self.change_rate = change_rate

    # Генерує рівномірні похибки в заданих межах.
    def get_measurement_error(self):
        errors = np.random.uniform(self.lowest_error_border, self.highest_error_border, self.size)
        return errors

    # Симулює постійний процес з додаванням випадкової похибки.
    def simulate_constant_process(self):
        values = [self.constant_value]
        for i in range(1, self.size):
            next_value = values[i - 1] + self.change_rate  # Зміна процесу на кожному кроці
            observed_value = next_value
            values.append(observed_value)
        return np.array(values)

    # Генерує експериментальні дані як суму двох моделей.
    def generate_experimental_data(self):
        constant_process = self.simulate_constant_process()
        measurement_error = self.get_measurement_error()
        additive_data = constant_process + measurement_error
        return additive_data

    # Обчислює середнє значення.
    def get_mean_value(self):
        mean = []
        for _ in range(self.size):
            experimental_data = self.generate_experimental_data()
            mean.append(np.mean(experimental_data))

        return np.mean(mean)

    # Обчислює дисперсію для експериментальних даних
    def get_dispersion(self):
        variance = []
        for _ in range(self.size):
            experimental_data = self.generate_experimental_data()
            variance.append(np.var(experimental_data))

        return np.mean(variance)

    # Обчислює середнє стандартне відхилення.
    def get_standard_deviation(self):
        standard_deviation = []
        for _ in range(self.size):
            experimental_data = self.generate_experimental_data()
            standard_deviation.append(np.std(experimental_data))

        return np.mean(standard_deviation)

    def generate_graph_of_additive_model(self):
        data = self.generate_experimental_data()
        constant_changes = np.cumsum(np.full(self.size, self.change_rate)) + self.constant_value

        plt.plot(data, color='b', label="Дані")
        plt.axhline(y=self.constant_value, color='r', linestyle='--', label="Значення константи")
        plt.plot(constant_changes, color='g', label="Дійсне значення із урахування зміни")

        plt.xlabel('Номер вимірювання')
        plt.ylabel('Значення')
        plt.title('Адитивна модель')
        plt.legend()
        plt.show()

    def generate_graph_of_monte_carlo_results(self):
        means = []
        for _ in range(self.size):
            experimental_data = self.generate_experimental_data()
            means.append(np.mean(experimental_data))

        plt.hist(means, bins=self.size, color='b', edgecolor='black')
        plt.title('Розподіл середніх значень Монте-Карло')
        plt.xlabel('Середнє значення')
        plt.ylabel('Частота')
        plt.show()

    # Генерує гістограму для експериментальних даних
    def generate_graph_of_data_distribution(self):
        data = self.generate_experimental_data()
        plt.hist(data, bins=self.size, color='g', edgecolor='black')
        plt.title('Розподіл даних')
        plt.xlabel('Значення')
        plt.ylabel('Частота')
        plt.show()

    # Генерує гістограму розподілу похибки
    def plot_error_distribution(self):
        errors = self.get_measurement_error()
        plt.hist(errors, bins=self.size, color='b', edgecolor='black')
        plt.title('Розподіл похибки')
        plt.xlabel('Похибка')
        plt.ylabel('Частота')
        plt.show()

    def get_mse_of_model(self):
        observed_data = self.generate_experimental_data()
        predicted_data = self.simulate_constant_process()
        mse = np.mean(np.square(observed_data - predicted_data))
        return mse

if __name__ == '__main__':
    model = AdditiveModel(lowest_error_border=-5,
                          highest_error_border=5,
                          size=30,
                          constant_value=10,
                          change_rate=2)

    mean_of_means = model.get_mean_value()
    mean_of_standard_deviation = model.get_standard_deviation()
    dispersion = model.get_dispersion()

    print(f"Середнє значення: {mean_of_means:.2f}")
    print(f"Середнє стандартне відхилення: {mean_of_standard_deviation:.2f}")
    print(f"Дисперсія (міра розподілу похибки) : {dispersion:.2f}")

    model.generate_graph_of_additive_model()
    model.generate_graph_of_monte_carlo_results()
    model.generate_graph_of_data_distribution()
    model.plot_error_distribution()
