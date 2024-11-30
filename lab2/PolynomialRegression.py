import numpy as np
from matplotlib import pyplot as plt

from lab1.AdditiveModel import AdditiveModel


class PolynomialRegression:
    def __init__(self):
        self.degree = 3
        self.coefficients = None

    def learn_using_lsm(self, x_axis, y_axis):
        """
        Навчання моделі за методом найменших квадратів.
        x_axis - Вхідні дані (незалежна змінна)
        y_axis - Цільові дані (залежна змінна)
        x_poly - матриця поліноміальних ознак
        Формула МНК:  = (X.T * X) ^ (-1) * X.T * y
        """
        x_poly = self.create_polynomial_matrix(x_axis)
        x_poly_transposed = x_poly.T
        x_poly_product = x_poly_transposed @ x_poly

        # Обчислення оберненої матриці (X.T @ X)^(-1)
        x_poly_inverse = np.linalg.inv(x_poly_product)
        x_poly_target_product = x_poly_transposed @ y_axis

        # Обчислення коефіцієнтів за формулою МНК
        self.coefficients = x_poly_inverse @ x_poly_target_product

    def predict(self, x_axis):
        if self.coefficients is None:
            raise ValueError("Модель ще не навчена. Використовуйте метод learn_using_lsm()")

        x_poly = self.create_polynomial_matrix(x_axis)
        return x_poly @ self.coefficients

    def create_polynomial_matrix(self, x_axis):
        """
        Генерація матриці поліноміальних ознак для заданого ступеня.
        x_axis - Вхідні дані
        """
        x_poly = np.vander(x_axis, self.degree + 1, increasing=True)
        return x_poly

    def extrapolate(self, x_axis, interval_ratio=0.5):
        size = len(x_axis)
        extension_size = int(size * interval_ratio)
        x_extended = np.arange(1, size + extension_size + 1)  # Розширений x_axis

        y_extrapolation = self.predict(x_extended)
        return x_extended, y_extrapolation

    def show_extrapolation(self, x_axis, y_axis, y_axis_prediction, x_extended=None, y_extrapolation=None):
        plt.figure(figsize=(10, 6))
        plt.scatter(x_axis, y_axis, color='grey', label='Дані')
        plt.plot(x_axis, y_axis_prediction, color='g', linewidth=5, label=f'Поліном 3-го ступеня')
        if x_extended is not None and y_extrapolation is not None:
            plt.plot(x_extended, y_extrapolation, color='g', linestyle='--', linewidth=5, label='Екстраполяція')
        plt.xlabel('x_axis')
        plt.ylabel('y_axis')
        plt.title('Екстраполяція параметрів досліджуваного процесу')
        plt.legend()
        plt.grid(True)
        plt.show()


if __name__ == '__main__':
    additive_model = AdditiveModel(lowest_error_border=-150,
                                   highest_error_border=200,
                                   size=100,
                                   constant_value=30,
                                   change_rate=20)

    x_axis = np.arange(1, additive_model.size + 1)
    y_axis = additive_model.generate_experimental_data()

    polynomial_model = PolynomialRegression()
    polynomial_model.learn_using_lsm(x_axis, y_axis)
    y_axis_prediction = polynomial_model.predict(x_axis)

    polynomial_model.show_extrapolation(x_axis, y_axis, y_axis_prediction)

    x_extended, y_axis_extrapolation = polynomial_model.extrapolate(x_axis)
    polynomial_model.show_extrapolation(x_axis, y_axis, y_axis_prediction, x_extended, y_axis_extrapolation)
