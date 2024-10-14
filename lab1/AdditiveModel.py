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

    # Генерує експериментальні дані як суму двох моделей (випадкової і невипадкової).
    def generate_experimental_data(self):
        constant_process = self.simulate_constant_process()
        measurement_error = self.get_measurement_error()
        additive_data = constant_process + measurement_error
        return additive_data

    def generate_graph(self):
        data = self.generate_experimental_data()
        constant_changes = np.cumsum(np.full(self.size, self.change_rate)) + self.constant_value

        plt.plot(data, color='b', label="Експериментальні дані")
        plt.axhline(y=self.constant_value, color='r', linestyle='--', label="Дійсне значення")
        plt.plot(constant_changes, color='g', label="Змінюване ідеальне значення")

        plt.xlabel('Номер вимірювання')
        plt.ylabel('Значення')
        plt.title('Адитивна модель експериментальних даних')
        plt.legend()
        plt.show()


if __name__ == '__main__':
    model = AdditiveModel(lowest_error_border=-5,
                          highest_error_border=5,
                          size=30,
                          constant_value=10,
                          change_rate=2)
    model.generate_graph()
