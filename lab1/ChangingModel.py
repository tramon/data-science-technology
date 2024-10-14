import numpy as np
import matplotlib.pyplot as plt


# Модель зміни досліджуваного процесу
class ChangingProcessWithUniformError:
    def __init__(self, initial_value, lowest_error_border, highest_error_border, size, change_rate):
        """
        initial_value - початкове значення.
        lowest_error_border - Нижня межа рівномірного розподілу похибки.
        highest_error_border - Верхня межа рівномірного розподілу похибки.
        size - Кількість точок для генерації.
        constant_value - Постійне значення процесу.
        change_rate - Швидкість зміни процесу (якщо позитивне - зростає, якщо негативне - спадає).
        """
        self.initial_value = initial_value
        self.lowest_error_border = lowest_error_border
        self.highest_error_border = highest_error_border
        self.size = size
        self.change_rate = change_rate

    # Генерує рівномірні похибки в заданих межах.
    def get_measurement_error(self):
        errors = np.random.uniform(self.lowest_error_border, self.highest_error_border, self.size)
        return errors

    # Функція для симуляції процесу, що змінюється з кожним кроком, з додаванням похибок.
    def simulate_changing_process(self):
        values = [self.initial_value]

        # Додавання похибки
        for i in range(1, self.size):
            next_value = values[i - 1] + self.change_rate  # Зміна процесу на кожному кроці
            observed_value = next_value + np.random.uniform(self.lowest_error_border, self.highest_error_border)
            values.append(observed_value)
        return np.array(values)

    # Генерує графік для візуалізації результатів процесу.
    def generate_graph(self):
        data = self.simulate_changing_process()
        plt.plot(data, label="Значення із похибкою")
        plt.plot([0, self.size - 1], [data[0], data[-1]], color='r', linestyle='--', label="Напрямок руху")
        plt.plot([0, self.size - 1], [data[0], np.mean(data)], color='g', linestyle='--',
                 label="Вектор середнього значення")
        plt.xlabel('Номер вимірювання')
        plt.ylabel('Значення')
        plt.title('Модель зміни досліджуваного процесу з рівномірною похибкою')
        plt.legend()
        plt.show()


if __name__ == '__main__':
    model = ChangingProcessWithUniformError(initial_value=0,
                                            lowest_error_border=-5,
                                            highest_error_border=5,
                                            size=30,
                                            change_rate=2)
    observed_data = model.simulate_changing_process()
    model.generate_graph()
