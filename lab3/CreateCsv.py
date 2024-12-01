import pandas as pd

if __name__ == '__main__':
    data = {
        "brand": ["Apple", "Dell", "HP", "Asus", "Lenovo", "MSI", "Acer", "Microsoft"],
        "model": [
            "MacBook Pro M1", "XPS 13", "Spectre x360", "ZenBook 14",
            "ThinkPad X1", "Prestige 15", "Swift 5", "Surface Laptop 4"
        ],
        "processor": [
            "Apple M1", "Intel i7-1165G7", "Intel i5-1135G7", "AMD Ryzen 5 5600U",
            "Intel i7-1260P", "Intel i7-1185G7", "Intel i5-1240P", "AMD Ryzen 7 4980U"
        ],
        "ram": [16, 16, 8, 8, 16, 16, 8, 16],
        "ssd": [512, 256, 512, 256, 1024, 512, 512, 256],
        "battery_life": [10, 12, 11, 9, 13, 7, 10, 11],
        "weight": [1.4, 1.2, 1.3, 1.2, 1.1, 1.6, 1.2, 1.3],
        "screen_size": [13.3, 13.4, 13.5, 14.0, 14.0, 15.6, 14.0, 13.5],
        "refresh_rate": [60, 60, 60, 90, 60, 120, 60, 60],
        "usb_ports": [2, 3, 3, 3, 2, 4, 3, 3],
        "gpu_score": [8, 7, 6, 7, 9, 8, 6, 7],
        "price": [1200, 1400, 1100, 1000, 1800, 1600, 900, 1300],
        "user_rating": [4.8, 4.5, 4.6, 4.4, 4.7, 4.3, 4.2, 4.5],
        "release_year": [2021, 2022, 2021, 2022, 2023, 2022, 2023, 2022],
    }

    laptops_with_weight_df = pd.DataFrame(data)

    weights = {
        "ram": 0.2,
        "ssd": 0.15,
        "battery_life": 0.15,
        "weight": 0.1,
        "refresh_rate": 0.1,
        "gpu_score": 0.1,
        "price": 0.1,
        "user_rating": 0.1,
        "release_year": 0.1
    }

    weights_row = pd.DataFrame([weights])
    weights_row["brand"] = "weights"
    weights_row["model"] = "weights"
    laptops_with_weight_df = pd.concat([laptops_with_weight_df, weights_row], ignore_index=True)

    # Збереження у CSV
    file_path = "laptops_with_weight.csv"
    laptops_with_weight_df.to_csv(file_path, index=False)
