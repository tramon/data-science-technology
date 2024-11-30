from lab1.AdditiveModel import AdditiveModel
from lab2 import DataClipper


class OptimizedAdditiveModel(AdditiveModel):
    def generate_experimental_data(self):
        constant_process = self.simulate_constant_process()
        measurement_error = self.get_measurement_error()
        additive_data = constant_process + measurement_error

        lower_bound = self.lowest_error_border * 0.75
        upper_bound = self.highest_error_border * 0.75
        clipped_data = DataClipper.clip(additive_data, lower_bound, upper_bound)

        return clipped_data


if __name__ == '__main__':
    base_model = AdditiveModel(lowest_error_border=-150,
                               highest_error_border=200,
                               size=30,
                               constant_value=30,
                               change_rate=2)

    base_data = base_model.generate_experimental_data()
    print("Базові дані:", base_data)
    mse_base = base_model.get_mse_of_model()
    print(f"MSE базової моделі: {mse_base:.2f}")

    optimized_model = OptimizedAdditiveModel(lowest_error_border=-150,
                                             highest_error_border=200,
                                             size=30,
                                             constant_value=30,
                                             change_rate=2)

    optimized_data = optimized_model.generate_experimental_data()

    print("\nОптимізовані дані:", optimized_data)
    mse_optimized = optimized_model.get_mse_of_model()
    print(f"MSE оптимізованої моделі: {mse_optimized:.2f}")
    print(f"\nРізниця MSE базової і MSE оптимізованої моделей: {mse_base - mse_optimized:.2f}")
