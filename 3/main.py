import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.statespace.sarimax import SARIMAX
from datetime import datetime, timedelta
from sklearn.metrics import mean_squared_error
from statsmodels.tsa.seasonal import seasonal_decompose
import mysql.connector
import warnings

warnings.filterwarnings('ignore')


def system_health_check():
    health_status = {"status": "healthy", "issues": []}

    required_libraries = [
        "pandas", "numpy", "matplotlib", "statsmodels",
        "sklearn", "mysql.connector"
    ]

    for lib in required_libraries:
        try:
            __import__(lib)
        except ImportError:
            health_status["status"] = "unhealthy"
            health_status["issues"].append(f"Бібліотека {lib} не встановлена")

    try:
        with open("test_write_access.tmp", "w") as f:
            f.write("test")
        import os
        os.remove("test_write_access.tmp")
    except (IOError, PermissionError):
        health_status["status"] = "unhealthy"
        health_status["issues"].append("Відсутній доступ до файлової системи")

    try:
        import mysql.connector
        conn = mysql.connector.connect(
            host="localhost",
            user="root",
            password="zaq1xsw2cde3#Edik",
            database="weather_db",
            port=330
        )
        conn.close()
    except Exception as e:
        health_status["status"] = "unhealthy"
        health_status["issues"].append(f"Помилка підключення до MySQL: {str(e)}")

    if health_status["status"] == "healthy":
        print("Системні перевірки пройдено успішно")
    else:
        print("Виявлено проблеми:")
        for issue in health_status["issues"]:
            print(f"  - {issue}")

    return health_status["status"] == "healthy"


def data_sanity_check(df, metadata):
    data_valid = True
    issues = []

    if len(df) == 0:
        data_valid = False
        issues.append("Датасет порожній")
        return False, issues

    if not pd.api.types.is_datetime64_dtype(df.index):
        data_valid = False
        issues.append("Індекс не є типом datetime")

    if not pd.api.types.is_numeric_dtype(df["temperature"]):
        data_valid = False
        issues.append("Температура не є числовим типом")

    missing_values = df["temperature"].isna().sum()
    if missing_values > 0:
        missing_percent = (missing_values / len(df)) * 100
        if missing_percent > 10:
            data_valid = False
            issues.append(f"Критична кількість пропущених значень: {missing_values} ({missing_percent:.2f}%)")
        else:
            print(f"Виявлено {missing_values} пропущених значень ({missing_percent:.2f}%)")

    extremely_cold = df[df["temperature"] < -50].shape[0]
    extremely_hot = df[df["temperature"] > 50].shape[0]

    if extremely_cold > 0:
        print(f"Виявлено {extremely_cold} записів з аномальною температурою (<-50°C)")
        if extremely_cold > 10:
            data_valid = False
            issues.append("Забагато записів з аномальною температурою")

    if extremely_hot > 0:
        print(f"Виявлено {extremely_hot} записів з аномальною температурою (>50°C)")
        if extremely_hot > 10:
            data_valid = False
            issues.append("Забагато записів з аномальною температурою")

    temp_mean = df["temperature"].mean()
    temp_std = df["temperature"].std()
    temp_min = df["temperature"].min()
    temp_max = df["temperature"].max()

    print(f"Статистика температур:")
    print(f"   Середня: {temp_mean:.2f}°C")
    print(f"   Мін/Макс: {temp_min:.2f}°C / {temp_max:.2f}°C")
    print(f"   Стандартне відхилення: {temp_std:.2f}°C")

    try:
        time_diff = df.index.to_series().diff().dropna()
        expected_diff = pd.Timedelta(time_diff.value_counts().idxmax())
        expected_mins = expected_diff.total_seconds() / 60

        large_gaps = time_diff[time_diff > expected_diff * 2]

        if len(large_gaps) > 0:
            print(f"Виявлено {len(large_gaps)} пропусків у часовому ряді")
            if len(large_gaps) < 5:
                for idx, gap in large_gaps.items():
                    gap_hours = gap.total_seconds() / 3600
                    print(f"   - Пропуск {gap_hours:.1f} годин починаючи з {idx}")

            if len(large_gaps) > len(df) * 0.05:
                data_valid = False
                issues.append("Забагато пропусків у часовому ряді")
    except Exception as e:
        print(f"Не вдалося проаналізувати часовий ряд: {str(e)}")

    date_range = df.index.max() - df.index.min()
    years = date_range.days / 365.25
    print(f"Датасет містить дані за {years:.1f} років ({date_range.days} днів)")
    if years < 1:
        print("Датасет містить дані менше ніж за 1 рік")
        print("Прогнозування сезонних трендів може бути ненадійним")

    if data_valid:
        print(f"Дані пройшли базові перевірки. Всього записів: {len(df)}")
    else:
        print("Виявлено критичні проблеми з даними:")
        for issue in issues:
            print(f"  - {issue}")

    return data_valid, issues


def model_validation_check(model, train, test):
    validation_passed = True
    issues = []
    try:
        residuals = pd.Series(model.resid)
        residual_mean = residuals.mean()
        print(f"Середнє між фактичними значеннями та прогнозованими значеннями моделі.: {residual_mean:.4f}")
        if abs(residual_mean) > 1.0:
            validation_passed = False
            issues.append(f"Середнє моделі сильно зміщені: {residual_mean:.4f}")

        plt.figure(figsize=(15, 5))
        plt.subplot(1, 2, 1)
        plt.hist(residuals, bins=30)
        plt.title("Розподіл дельти")
        plt.subplot(1, 2, 2)
        plt.scatter(range(len(residuals)), residuals)
        plt.axhline(y=0, color='r', linestyle='-')
        plt.title("Графік дельти")
        plt.tight_layout()
        plt.savefig('model_residuals.png')
    except Exception as e:
        print(f"Не вдалося проаналізувати: {str(e)}")

    if validation_passed:
        print("Модель пройшла валідацію")
    else:
        print("Виявлено проблеми з моделлю:")
        for issue in issues:
            print(f"  - {issue}")

    return validation_passed, issues


def forecast_sanity_check(forecast_df):
    forecast_valid = True
    issues = []
    if forecast_df["temperature"].isna().any():
        forecast_valid = False
        issues.append("Прогноз містить пропущені значення")

    min_temp = forecast_df["temperature"].min()
    max_temp = forecast_df["temperature"].max()
    if min_temp < -50 or max_temp > 50:
        forecast_valid = False
        issues.append(f"Прогноз містить нереалістичні значення температури: від {min_temp:.1f}°C до {max_temp:.1f}°C")

    temp_diff = forecast_df["temperature"].diff().abs()
    max_diff = temp_diff.max()
    if max_diff > 15:
        forecast_valid = False
        issues.append(f"Виявлено різкий стрибок температури: {max_diff:.1f}°C")

    if forecast_valid:
        print("Прогноз пройшов перевірку логічності")
    else:
        print("Виявлено проблеми з прогнозом:")
        for issue in issues:
            print(f"  - {issue}")

    return forecast_valid, issues


def database_check():
    db_valid = True
    issues = []

    try:
        conn = mysql.connector.connect(
            host="localhost",
            user="root",
            password="zaq1xsw2cde3#Edik",
            database="weather_db",
            port=330
        )
        cursor = conn.cursor()


        try:
            cursor.execute("INSERT INTO forecast (date, temperature) VALUES (CURDATE(), 0)")
            cursor.execute("DELETE FROM forecast WHERE temperature = 0 AND date = CURDATE()")
            conn.commit()
        except Exception as e:
            db_valid = False
            issues.append(f"Відсутні права на запис в БД: {str(e)}")
        cursor.close()
        conn.close()
    except Exception as e:
        db_valid = False
        issues.append(f"Помилка роботи з БД: {str(e)}")

    if not db_valid:
        print("Виявлено проблеми з базою даних:")
        for issue in issues:
            print(f"  - {issue}")
    return db_valid, issues

def load_data(file_path):
    metadata = {}
    with open(file_path, 'r') as f:
        for i in range(8):
            line = f.readline().strip().split(',')
            if len(line) >= 2:
                metadata[line[0]] = line[1]

    df = pd.read_csv(file_path, skiprows=9)
    df.columns = ['timestamp', 'temperature']
    df['timestamp'] = pd.to_datetime(df['timestamp'], format    ='%Y%m%dT%H%M')
    df.set_index('timestamp', inplace=True)
    return df, metadata

def analyze_time_series(df):
    daily_data = df.resample('D').mean()
    monthly_data = df.resample('M').mean()

    if len(monthly_data) >= 24:
        decomposition = seasonal_decompose(monthly_data, model='additive', period=12)
        plt.figure(figsize=(20, 10))
        plt.subplot(411)
        plt.plot(decomposition.observed)
        plt.title('Спостережені дані')
        plt.subplot(412)
        plt.plot(decomposition.trend)
        plt.title('Тренд')
        plt.subplot(413)
        plt.plot(decomposition.seasonal)
        plt.title('Сезонність')
        plt.subplot(414)
        plt.plot(decomposition.resid)
        plt.title('Залишки')
        plt.tight_layout()
        plt.savefig('seasonal_decomposition.png')
    else:
        print(f"Недостатньо даних для декомпозиції: {len(monthly_data)} < 24 місяців")

    return daily_data, monthly_data

def train_model(data):
    train_size = int(len(data) * 0.8)
    train, test = data[:train_size], data[train_size:]
    best_rmse = float('inf')
    best_params = None
    best_model = None

    p_values = range(0, 3)
    d_values = range(0, 2)
    q_values = range(0, 3)
    P_values = range(0, 2)
    D_values = range(0, 2)
    Q_values = range(0, 2)

    for p in p_values:
        for d in d_values:
            for q in q_values:
                for P in P_values:
                    for D in D_values:
                        for Q in Q_values:
                            try:
                                model = SARIMAX(train, order=(p, d, q),
                                                seasonal_order=(P, D, Q, 12))
                                model_fit = model.fit(disp=False)
                                predictions = model_fit.forecast(steps=len(test))
                                rmse = np.sqrt(mean_squared_error(test, predictions))

                                if rmse < best_rmse:
                                    best_rmse = rmse
                                    best_params = (p, d, q, P, D, Q)
                                    best_model = model_fit

                                print(f"ARIMA({p},{d},{q})({P},{D},{Q},12) RMSE: {rmse:.2f}")
                            except:
                                continue

    print(f"Найкращі параметри: ARIMA{best_params[:3]}({best_params[3:]},12)")
    print(f"Найкращий RMSE: {best_rmse:.2f}")

    predictions = best_model.forecast(steps=len(test))
    plt.figure(figsize=(20, 7))
    plt.plot(train.index, train, label='Тренувальні')
    plt.plot(test.index, test, label='Тестові')
    plt.plot(test.index, predictions, color='red', label='Прогноз')
    plt.title('SARIMA - Фактичні vs Прогноз')
    plt.xlabel('Дата')
    plt.ylabel('Температура')
    plt.legend()
    plt.grid(True)
    plt.savefig('model_validation.png')

    return best_model

def save_forecast_to_mysql(forecast_df):
    conn = mysql.connector.connect(
        host="localhost",
        user="root",
        password="zaq1xsw2cde3#Edik",
        database="weather_db",
        port = 330

    )
    cursor = conn.cursor()

    cursor.execute("DELETE FROM forecast")

    for _, row in forecast_df.iterrows():
        cursor.execute(
            "INSERT INTO forecast (date, temperature) VALUES (%s, %s)",
            (row['date'].date(), float(row['temperature']))
        )

    conn.commit()
    cursor.close()
    conn.close()
    print("Прогноз записано в MySQL.")


def read_forecast_from_mysql():
    conn = mysql.connector.connect(
        host="localhost",
        user="root",
        password="zaq1xsw2cde3#Edik",
        database="weather_db",
        port = 330
    )
    df = pd.read_sql("SELECT date, temperature FROM forecast ORDER BY date", conn)
    conn.close()
    print("\nПрогноз на 20 років з бази:")
    print(df.head(20))

    return df


def forecast_future(model, last_date, metadata):

    months = 12 * 20
    forecast = model.forecast(steps=months)
    start_date = last_date + timedelta(days=1)
    future_dates = pd.date_range(start=start_date, periods=months, freq='M')
    base_forecast = forecast.copy()

    trend_factor = 0.015
    trend_acceleration = 0.0001
    trend = np.array([trend_factor * i + trend_acceleration * (i ** 2) for i in range(months)])

    noise = np.random.normal(0, 0.5, months) * np.sqrt(np.arange(1, months + 1) / 10)

    anomalies = np.zeros(months)
    n_anomalies = np.random.randint(3, 6)
    for _ in range(n_anomalies):
        pos = np.random.randint(12, months - 12)
        intensity = np.random.choice([-1, 1]) * np.random.uniform(1.0, 2.5)
        duration = np.random.randint(3, 13)
        for i in range(duration):
            weight = 0.5 * (1 - np.cos(2 * np.pi * i / duration))
            if pos + i < months:
                anomalies[pos + i] += intensity * weight

    forecast_with_variability = base_forecast + trend + noise + anomalies

    forecast_df = pd.DataFrame({
        'date': future_dates,
        'base_forecast': base_forecast,
        'forecast_with_trend': base_forecast + trend,
        'temperature': forecast_with_variability
    })


    plt.figure(figsize=(40, 7))
    plt.plot(forecast_df['date'], forecast_df['base_forecast'],
             label='Базовий прогноз', alpha=0.6, color='blue')
    plt.plot(forecast_df['date'], forecast_df['forecast_with_trend'],
             label='Прогноз з трендом', alpha=0.6, color='green')
    plt.plot(forecast_df['date'], forecast_df['temperature'],
             label='Прогноз з варіативністю', color='red')
    plt.title(f"Прогноз температури на 20 років вперед ({metadata.get('location', 'невідомо')})")
    plt.xlabel('Дата')
    plt.ylabel('Температура (°C)')
    plt.legend()
    plt.grid(True)
    plt.savefig('20_year_forecast.png')

    plt.figure(figsize=(20, 7))
    plt.plot(forecast_df['date'], forecast_df['temperature'], label='Прогноз', color='red')

    conf_interval = 1.0 + np.sqrt(np.arange(months) / 100)
    plt.fill_between(forecast_df['date'],
                     forecast_df['temperature'] - conf_interval,
                     forecast_df['temperature'] + conf_interval,
                     alpha=0.2, color='red', label='95% довірчий інтервал')

    plt.title(f"Прогноз температури з довірчими інтервалами ({metadata.get('location', 'невідомо')})")
    plt.xlabel('Дата')
    plt.ylabel('Температура (°C)')
    plt.legend()
    plt.grid(True)
    plt.savefig('forecast_with_intervals.png')

    return forecast_df


def main(file_path):
    all_issues = {
        'data_issues': [],
        'model_issues': [],
        'forecast_issues': [],
        'db_issues': []
    }

    print("\n=== ЗАПУСК СИСТЕМНИХ ПЕРЕВІРОК ===")
    system_ok = system_health_check()
    if not system_ok:
        print("Виявлено системні проблеми, але спробуємо продовжити виконання")

    print("\n=== ЗАВАНТАЖЕННЯ CSV ===")
    try:
        df, metadata = load_data(file_path)
        print(f"Дані для {metadata.get('location', 'невідомої локації')}")
        print(f"Період: з {df.index.min()} по {df.index.max()}")
        print(f"Кількість записів: {len(df)}")
    except Exception as e:
        print(f"Помилка завантаження даних: {str(e)}")
        return

    print(f"\n=== ПЕРЕВІРКА ДАНИХ ===")
    data_valid, data_issues = data_sanity_check(df, metadata)
    all_issues['data_issues'] = data_issues

    if not data_valid:
        print("Виявлено проблеми з даними.")
        user_response = input("Бажаєте продовжити аналіз? (y/n): ")
        if user_response.lower() != 'y':
            return

    print(f"\n=== АНАЛІЗ ЧАСОВОГО РЯДУ ===")
    daily_data, monthly_data = analyze_time_series(df)

    print("\n=== ПЕРЕВІРКА БАЗИ ДАНИХ ===")
    db_valid, db_issues = database_check()
    all_issues['db_issues'] = db_issues

    if not db_valid:
        print("Проблеми з базою даних. Збереження результатів буде пропущено.")

    print("\n=== ТРЕНУВАННЯ МОДЕЛІ ===")
    train_size = int(len(monthly_data) * 0.8)
    train, test = monthly_data[:train_size], monthly_data[train_size:]

    try:
        model = train_model(monthly_data)

        print("\n=== ВАЛІДАЦІЯ МОДЕЛІ ===")
        model_valid, model_issues = model_validation_check(model, train, test)
        all_issues['model_issues'] = model_issues

        if not model_valid:
            print("Виявлено проблеми з моделлю. Прогноз може бути ненадійним.")
            user_response = input("Бажаєте продовжити? (y/n): ")
            if user_response.lower() != 'y':
                return

        print("\n=== ГЕНЕРАЦІЯ ПРОГНОЗУ ===")
        forecast_df = forecast_future(model, df.index.max(), metadata)

        print("\n=== ПЕРЕВІРКА ПРОГНОЗУ ===")
        forecast_valid, forecast_issues = forecast_sanity_check(forecast_df)
        all_issues['forecast_issues'] = forecast_issues

        if not forecast_valid:
            print("Виявлено проблеми з прогнозом. Результати можуть бути ненадійними.")
            user_response = input("Бажаєте зберегти прогноз? (y/n): ")
            if user_response.lower() != 'y':
                return

        if db_valid:
            print("\n=== ЗБЕРЕЖЕННЯ В БАЗУ ===")
            save_forecast_to_mysql(forecast_df)
            print("Прогноз збережено в базу даних")

            print("\n=== ЗЧИТУВАННЯ З БАЗИ ===")
            read_forecast_from_mysql()

    except Exception as e:
        print(f"Помилка під час аналізу та прогнозування: {str(e)}")
        import traceback
        traceback.print_exc()
        return

if __name__ == "__main__":
    main("temperature_data.csv")
