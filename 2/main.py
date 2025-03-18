import json
from datetime import datetime
from pymongo import MongoClient

client = MongoClient("mongodb://localhost:27017/")

db = client["KFC1"]

meters_collection = db["meters"]
billing_collection = db["billing"]

DAY_TARIFF = 2.5
NIGHT_TARIFF = 1.2
ROLLOVER_DAY = 100
ROLLOVER_NIGHT = 80




def load_meters():
    return list(meters_collection.find({}, {"_id": 0}))

def load_billing():
    return {bill["meter_id"]: bill["history"] for bill in billing_collection.find({}, {"_id": 0})}

def save_meter(meter):
    meters_collection.update_one({"id": meter["id"]}, {"$set": meter}, upsert=True)

def save_meter_reading(meter_id, day, night):
    reading = {
        "meter_id": meter_id,
        "date": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "corrected_day": day,
        "corrected_night": night
    }
    db["meter_history"].insert_one(reading)


def save_billing(meter_id, bill_record):
    billing_collection.update_one(
        {"meter_id": meter_id},
        {"$push": {"history": bill_record}},
        upsert=True
    )


def calculate_bill(meter_id, prev_day, prev_night, current_day, current_night):

    day_diff = current_day - prev_day
    night_diff = current_night - prev_night


    if day_diff < 0:
        day_diff = ROLLOVER_DAY
    if night_diff < 0:
        night_diff = ROLLOVER_NIGHT

    day_cost = day_diff * DAY_TARIFF
    night_cost = night_diff * NIGHT_TARIFF
    total_cost = day_cost + night_cost

    bill_record = {
        "date": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "day_consumption": day_diff,
        "night_consumption": night_diff,
        "day_cost": day_cost,
        "night_cost": night_cost,
        "total_cost": total_cost
    }

    return bill_record


def process_meter_reading(meter_id, current_day, current_night):

    meter = meters_collection.find_one({"id": meter_id}, {"_id": 0})

    if not meter:
        prev_day = 0
        prev_night = 0
        meter = {
            "id": meter_id,
            "previous_day": current_day,
            "previous_night": current_night,
            "last_updated": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        }
        save_meter(meter)

        bill = calculate_bill(meter_id, prev_day, prev_night, current_day, current_night)
        save_billing(meter_id, bill)

        save_meter_reading(meter_id, current_day, current_night)

        return meter, bill

    prev_day = meter["previous_day"]
    prev_night = meter["previous_night"]

    if current_day < prev_day:
        current_day = prev_day + ROLLOVER_DAY
    if current_night < prev_night:
        current_night = prev_night + ROLLOVER_NIGHT

    bill = calculate_bill(meter_id, prev_day, prev_night, current_day, current_night)

    updated_meter = {
        "id": meter_id,
        "previous_day": current_day,
        "previous_night": current_night,
        "last_updated": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    }
    save_meter(updated_meter)
    save_billing(meter_id, bill)

    save_meter_reading(meter_id, current_day, current_night)

    return updated_meter, bill


def get_meter_history(meter_id):
    history = db["meter_history"].find({"meter_id": meter_id}, {"_id": 0})
    return list(history)

def get_billing_history(meter_id):
    bill = billing_collection.find_one({"meter_id": meter_id}, {"_id": 0})
    return bill["history"] if bill else []


def main():
    print("Програма обліку електроенергії")

    while True:
        print("\nМеню:")
        print("1. Ввести нові показники")
        print("2. Переглянути історію показників")
        print("3. Переглянути історію платежів")
        print("0. Вийти")

        choice = input("Виберіть дію: ")
        if choice == "0":
            print("Програма завершена.")
            break
        if "0" <=  choice <= "3":
            meter_id = input("\nВведіть ID лічильника: ")
        if choice == "1":
            try:
                current_day = float(input("Введіть поточні показники (день): "))
                current_night = float(input("Введіть поточні показники (ніч): "))
            except ValueError:
                print("Помилка: введіть числові значення.")
                continue

            meter = meters_collection.find_one({"id": meter_id}, {"_id": 0})

            if meter:
                prev_day = meter["previous_day"]
                prev_night = meter["previous_night"]

                if current_day < prev_day or current_night < prev_night:
                    print("\nУвага! Ви ввели показники менші за попередні.")
                    confirm = input("Ви хочете підтвердити накрутку (100 день, 80 ніч)? (y/n): ")
                    if confirm.lower() != "y":
                        print("Введіть показники ще раз.")
                        continue

            meter, bill = process_meter_reading(meter_id, current_day, current_night)

            print("\nРезультати обробки:")
            print(f"Лічильник ID: {meter['id']}")
            print(f"Оновлені показники (день): {meter['previous_day']} кВт")
            print(f"Оновлені показники (ніч): {meter['previous_night']} кВт")
            print(f"Загальна сума до сплати: {bill['total_cost']} грн")

        elif choice == "2":
            history = get_meter_history(meter_id)
            if history:
                print("\nІсторія показників:")
                for entry in history:
                    print(f"{entry['date']}: День {entry['corrected_day']} кВт, Ніч {entry['corrected_night']} кВт")
            else:
                print("Історія відсутня.")

        elif choice == "3":
            billing = get_billing_history(meter_id)
            if billing:
                print("\nІсторія платежів:")
                for entry in billing:
                    print(f"{entry['date']}: {entry['total_cost']} грн")
            else:
                print("Немає даних про платежі.")

        else:
            print("Невірний вибір. Спробуйте ще раз.")

if __name__ == "__main__":
    main()

