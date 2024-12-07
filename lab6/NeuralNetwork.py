import numpy as np
from matplotlib import pyplot as plt


class NeuralNetwork:
    def __init__(self, input_size, hidden_size, output_size, learning_rate=0.1):
        """
        input_size - Розмір вхідного шару
        hidden_size - Кількість нейронів у прихованому шарі
        output_size - Розмір вихідного шару
        learning_rate - Швидкість навчання
        """
        self.learning_rate = learning_rate

        # Ініціалізація ваг для вхідного та прихованого шару
        self.weights_input_hidden = np.random.uniform(-1, 1, (input_size, hidden_size))
        self.bias_hidden = np.random.uniform(-1, 1, (1, hidden_size))

        # Ініціалізація ваг для прихованого та вихідного шару
        self.weights_hidden_output = np.random.uniform(-1, 1, (hidden_size, output_size))
        self.bias_output = np.random.uniform(-1, 1, (1, output_size))

    @staticmethod
    def sigmoid(x):
        return 1 / (1 + np.exp(-x))

    @staticmethod
    def sigmoid_derivative(x):
        return x * (1 - x)

    def forward(self, inputs):
        # Обчислення значень прихованого шару
        self.hidden_layer_input = np.dot(inputs, self.weights_input_hidden) + self.bias_hidden
        self.hidden_layer_output = self.sigmoid(self.hidden_layer_input)

        # Обчислення виходу
        self.output_layer_input = np.dot(self.hidden_layer_output, self.weights_hidden_output) + self.bias_output
        self.output_layer_output = self.sigmoid(self.output_layer_input)

        return self.output_layer_output

    def backward(self, inputs, expected_output, predicted_output):

        # Помилка вихідного шару
        error = expected_output - predicted_output
        output_gradient = error * self.sigmoid_derivative(predicted_output)

        # Помилка прихованого шару
        hidden_error = np.dot(output_gradient, self.weights_hidden_output.T)
        hidden_gradient = hidden_error * self.sigmoid_derivative(self.hidden_layer_output)

        # Оновлення ваг та зміщень
        self.weights_hidden_output += np.dot(self.hidden_layer_output.T, output_gradient) * self.learning_rate
        self.bias_output += np.sum(output_gradient, axis=0, keepdims=True) * self.learning_rate

        self.weights_input_hidden += np.dot(inputs.T, hidden_gradient) * self.learning_rate
        self.bias_hidden += np.sum(hidden_gradient, axis=0, keepdims=True) * self.learning_rate

    def train(self, inputs, expected_output, epochs=10000):
        losses = []
        for epoch in range(epochs):
            predicted_output = self.forward(inputs)
            self.backward(inputs, expected_output, predicted_output)

            # Розрахунок помилки (loss)
            loss = np.mean((expected_output - predicted_output) ** 2)
            losses.append(loss)

            if epoch % 1000 == 0:
                print(f"Епоха {epoch}, Помилка: {loss:.4f}")

        plt.plot(losses, label="Помилка", linewidth=4, color='blue')
        plt.xlabel("Епоха")
        plt.ylabel("Помилка")
        plt.title("Зміна помилки під час навчання")
        plt.legend()
        plt.grid(True)
        plt.show()

    @staticmethod
    def plot(expected_output, predicted):
        plt.plot(expected_output, label="Очікувані значення", linestyle="-", marker="o", linewidth=3, color='orange')
        plt.plot(predicted, label="Передбачені значення", linestyle="--", marker="x", linewidth=3, color='blue')
        plt.xlabel("Індекс")
        plt.ylabel("Значення")
        plt.title("Очікувані vs Передбачені значення")
        plt.legend()
        plt.grid(True)
        plt.show()


if __name__ == '__main__':
    input = np.array([
        [0, 1, 1],
        [1, 1, 0],
        [1, 0, 0],
        [1, 1, 1],
        [0, 0, 1]
    ])

    expected_output = np.array([
        [0],
        [0],
        [1],
        [1],
        [1]
    ])

    nn = NeuralNetwork(input_size=3, hidden_size=4, output_size=1, learning_rate=0.1)
    nn.train(input, expected_output, epochs=10000)
    predicted = nn.forward(input)

    print(f"\nРезультати після навчання:\n {predicted}")

    NeuralNetwork.plot(expected_output, predicted)
