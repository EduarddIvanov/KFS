import unittest
from main import process_meter_reading, meters_collection, billing_collection

class TestMeterProcessing(unittest.TestCase):

    def setUp(self):
        meters_collection.delete_many({})
        billing_collection.delete_many({})

    def test_update_existing_meter(self):
        initial_data = {"id": "1234", "previous_day": 100, "previous_night": 50, "last_updated": "2025-01-01 12:00:00"}
        meters_collection.insert_one(initial_data)

        process_meter_reading("1234", 150, 70)

        meter = meters_collection.find_one({"id": "1234"}, {"_id": 0})
        self.assertEqual(meter["previous_day"], 150)
        self.assertEqual(meter["previous_night"], 70)

    def test_new_meter_entry(self):
        process_meter_reading("5678", 200, 100)

        meter = meters_collection.find_one({"id": "5678"}, {"_id": 0})
        self.assertIsNotNone(meter)
        self.assertEqual(meter["previous_day"], 200)
        self.assertEqual(meter["previous_night"], 100)

    def test_night_rollover(self):
        meters_collection.insert_one({"id": "9999", "previous_day": 300, "previous_night": 250, "last_updated": "2025-01-01 12:00:00"})

        process_meter_reading("9999", 320, 200)
        meter = meters_collection.find_one({"id": "9999"}, {"_id": 0})
        self.assertEqual(meter["previous_night"], 330)

    def test_day_rollover(self):

        meters_collection.insert_one({"id": "8888", "previous_day": 500, "previous_night": 300, "last_updated": "2025-01-01 12:00:00"})

        process_meter_reading("8888", 450, 320)

        meter = meters_collection.find_one({"id": "8888"}, {"_id": 0})
        self.assertEqual(meter["previous_day"], 600)

    def test_both_rollovers(self):
        meters_collection.insert_one({"id": "7777", "previous_day": 600, "previous_night": 400, "last_updated": "2025-01-01 12:00:00"})

        process_meter_reading("7777", 550, 350)
        meter = meters_collection.find_one({"id": "7777"}, {"_id": 0})
        self.assertEqual(meter["previous_day"], 700)
        self.assertEqual(meter["previous_night"], 480)

if __name__ == "__main__":
    unittest.main()
