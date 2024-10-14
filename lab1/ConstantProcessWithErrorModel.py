import numpy as np
import matplotlib.pyplot as plt


# Модель генерації випадкової величини постійного процесу з рівномірною похибкою
class ConstantProcessWithErrorModel:
    def __init__(self, lowest_error_border, highest_error_border, size, constant_value):
        """
        lowest_error_border - Нижня межа рівномірного розподілу похибки.
        highest_error_border - Верхня межа рівномірного розподілу похибки.
        size - Кількість точок для генерації.
        constant_value - Постійне значення процесу.
        """
        self.lowest_error_border = lowest_error_border
        self.highest_error_border = highest_error_border
        self.size = size
        self.constant_value = constant_value

    # Функція генерації похибки на основі рівномірного розподілу
    def get_measurement_error(self):
        errors = np.random.uniform(self.lowest_error_border, self.highest_error_border, self.size)
        return errors

    # Функція для симуляції постійного процесу з похибками
    def simulate_constant_process_with_error(self):
        errors = self.get_measurement_error()
        observed_values = self.constant_value + errors
        return observed_values

    # Функція генерації графіку
    def generate_graph(self):
        data = self.simulate_constant_process_with_error()

        plt.plot(data, label="Значення із похибкою")
        plt.axhline(y=self.constant_value, color='r', linestyle='-', label="Дійсне значення")
        plt.axhline(y=np.mean(data), color='g', linestyle='--', label="Середнє значення")

        plt.xlabel('Номер вимірювання')
        plt.ylabel('Значення')
        plt.title('Модель генерації випадкової величини постійного процесу з рівномірною похибкою')
        plt.legend()
        plt.show()


if __name__ == '__main__':
    model = ConstantProcessWithErrorModel(lowest_error_border=-5, highest_error_border=5, size=30, constant_value=10)
    observed_data = model.simulate_constant_process_with_error()
    model.generate_graph()
