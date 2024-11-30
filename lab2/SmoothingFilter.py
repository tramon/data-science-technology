import numpy as np
import matplotlib.pyplot as plt

from lab1.AdditiveModel import AdditiveModel


class SmoothingFilter:
    def __init__(self, alpha=0.8, beta=0.2, gamma=None):
        """
        alpha - коефіцієнт згладжування позиції
        beta - коефіцієнт згладжування швидкості
        """
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma

    def a_b_smoothing(self, data):
        """
        Повертає:
        - масив згладжених позицій
        - масив швидкостей
        """
        # Початкові оцінки
        position = data[0]
        velocity = 0

        smoothed_positions = [position]
        smoothed_velocities = [velocity]

        for k in range(1, len(data)):
            predicted_position = position + velocity
            residual = data[k] - predicted_position

            position = predicted_position + self.alpha * residual
            velocity = velocity + self.beta * residual

            smoothed_positions.append(position)
            smoothed_velocities.append(velocity)

        return np.array(smoothed_positions), np.array(smoothed_velocities)

    def plot_smoothing_results(self,
                               original_data,
                               smoothed_positions,
                               smoothed_velocities,
                               smoothed_accelerations=None):
        """
        original_data - оригінальні дані
        smoothed_positions - згладжені позиції
        smoothed_velocities - гладжені швидкості
        smoothed_accelerations - (опціонально) згладжені прискорення
        """
        plt.figure(figsize=(12, 6))
        plt.plot(original_data, label="Оригінальні дані (з шумом)", linestyle="--", color="grey")
        plt.plot(smoothed_positions, label="Згладжені позиції", linewidth=2, color="blue")
        plt.plot(smoothed_velocities, label="Згладжені швидкості", linewidth=2, color="orange")
        if smoothed_accelerations is not None:
            plt.plot(smoothed_accelerations, label="Згладжені прискорення", linewidth=2, color="green")
        plt.xlabel("Елемент")
        plt.ylabel("Значення")
        if smoothed_accelerations is None:
            plt.title("Рекурентне згладжування (α-β)")
        else:
            plt.title("Рекурентне згладжування (α-β-γ)")
        plt.legend()
        plt.grid(True)
        plt.show()

    def a_b_y_smoothing(self, data):
        """
        Повертає:
        - масив згладжених позицій
        - масив швидкостей
        - масив прискорень
        """

        position = data[0]
        velocity = 0
        acceleration = 0

        smoothed_positions = [position]
        smoothed_velocities = [velocity]
        smoothed_accelerations = [acceleration]

        for k in range(1, len(data)):
            predicted_position = position + velocity + 0.5 * acceleration
            residual = data[k] - predicted_position

            position = predicted_position + self.alpha * residual
            velocity = velocity + acceleration + self.beta * residual
            acceleration = acceleration + self.gamma * residual

            smoothed_positions.append(position)
            smoothed_velocities.append(velocity)
            smoothed_accelerations.append(acceleration)

        return np.array(smoothed_positions), np.array(smoothed_velocities), np.array(smoothed_accelerations)


if __name__ == '__main__':
    additive_model_rising = AdditiveModel(lowest_error_border=-150,
                                          highest_error_border=200,
                                          size=100,
                                          constant_value=0,
                                          change_rate=0)

    stale_rate_data = additive_model_rising.generate_experimental_data()

    a_b_smoothing = SmoothingFilter(alpha=0.6, beta=0.2)
    smoothed_positions, smoothed_velocities = a_b_smoothing.a_b_smoothing(stale_rate_data)
    a_b_smoothing.plot_smoothing_results(stale_rate_data, smoothed_positions, smoothed_velocities)

    additive_model_rising = AdditiveModel(lowest_error_border=-150,
                                          highest_error_border=200,
                                          size=100,
                                          constant_value=0,
                                          change_rate=20)
    rising_rate_data = additive_model_rising.generate_experimental_data()

    a_b_g_smoothing = SmoothingFilter(alpha=0.6, beta=0.2, gamma=0.2)
    a_b_g_positions, a_b_g_velocities, a_b_g_accelerations = a_b_g_smoothing.a_b_y_smoothing(rising_rate_data)

    a_b_smoothing.plot_smoothing_results(rising_rate_data,
                                         a_b_g_positions,
                                         a_b_g_velocities,
                                         a_b_g_accelerations)
