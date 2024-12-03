import numpy as np
import requests
import pandas as pd
import datetime

from matplotlib import pyplot as plt

from lab2.PolynomialRegression import PolynomialRegression

base_url = "https://api.privatbank.ua/p24api/exchange_rates?date={date}"
buy = "buy"
sale = "sale"
date_formatter = "%d.%m.%Y"


class CurrencyAnalysis:

    @staticmethod
    def fetch_usd_rate(days_to_fetch):
        historical_data = []

        for i in range(days_to_fetch):
            date = (datetime.datetime.now() - datetime.timedelta(days=i)).strftime(date_formatter)
            url = base_url.format(date=date)
            response = requests.get(url)

            if response.status_code == 200:
                data = response.json()
                exchange_rates = data.get("exchangeRate", [])
                usd_rate = None
                for rate in exchange_rates:
                    if rate.get("currency") == "USD":
                        usd_rate = rate
                        break

                if usd_rate:
                    historical_data.append({
                        "date": date,
                        buy: usd_rate.get("purchaseRate"),
                        sale: usd_rate.get("saleRate")
                    })
            else:
                print(f"Не вдалося отримати дані за {date}. Код: {response.status_code}")

        return pd.DataFrame(historical_data)[::-1]

    @staticmethod
    def predict_next_day(exchange_rates, rate_type=buy):
        if rate_type not in [buy, sale]:
            raise ValueError("rate_type має бути 'buy' або 'sale'")

        exchange_rates["date_index"] = np.arange(len(exchange_rates))
        x_axis = exchange_rates["date_index"].values
        y_axis = exchange_rates[rate_type].values

        polynomial_model = PolynomialRegression(degree=1)
        polynomial_model.learn_using_lsm(x_axis, y_axis)

        next_day_index = len(exchange_rates)
        predicted_rate = polynomial_model.predict(np.array([next_day_index]))
        return predicted_rate[0]

    @staticmethod
    def plot_exchange_rates_with_trend(exchange_rates, predicted_value, rate_type="buy"):
        if rate_type not in [buy, sale]:
            raise ValueError("rate_type має бути 'buy' або 'sale'")

        dates = list(exchange_rates["date"])
        rates = list(exchange_rates[rate_type])

        next_date = (max(pd.to_datetime(dates, dayfirst=True)) + pd.Timedelta(days=1)).strftime(date_formatter)
        dates.append(next_date)
        rates.append(predicted_value)

        plt.figure(figsize=(12, 6))
        plt.plot(dates[:-1], rates[:-1], label="Курс USD", linewidth=5, color="grey", marker="o")
        plt.plot(dates[-2:], rates[-2:], label="Прогнозований тренд", linewidth=5, color="green", linestyle="--",
                 marker="*")
        plt.scatter(next_date, predicted_value, color='Blue', linewidth=8, label=f"Прогноз: {predicted_value:.2f}",
                    zorder=5)
        plt.title(f"Графік курсу USD {rate_type} з прогнозом")
        plt.xlabel("Дата")
        plt.ylabel("Курс")
        plt.xticks(rotation=45)
        plt.grid(True)
        plt.legend()
        plt.tight_layout()
        plt.show()


if __name__ == '__main__':
    """
    Довго опрацьовує запити до pb.ua (декілька хвилин)
    В API pb.ua не передбачено отримання одним запитом списку курсів за період
    Документація по API pb.ua https://api.privatbank.ua/#p24/exchangeArchive
    """

    exchange_rates = CurrencyAnalysis.fetch_usd_rate(9)
    print(exchange_rates)

    buy_rate_prediction = CurrencyAnalysis.predict_next_day(exchange_rates, rate_type="buy")
    sale_rate_prediction = CurrencyAnalysis.predict_next_day(exchange_rates, rate_type="sale")

    print(f"Прогнозований курс купівлі на завтра: {buy_rate_prediction}")
    print(f"Прогнозований курс продажу на завтра: {sale_rate_prediction}")

    CurrencyAnalysis.plot_exchange_rates_with_trend(exchange_rates, buy_rate_prediction, rate_type="buy")
    CurrencyAnalysis.plot_exchange_rates_with_trend(exchange_rates, sale_rate_prediction, rate_type="sale")
