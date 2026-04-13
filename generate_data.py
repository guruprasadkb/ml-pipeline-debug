"""Generate synthetic fraud detection dataset for the ML challenge."""
import random
import csv
import os


def generate_dataset(n=1000, fraud_rate=0.05, seed=42):
    random.seed(seed)

    header = [
        "transaction_id", "amount", "merchant_category", "hour_of_day",
        "day_of_week", "distance_from_home", "distance_from_last_transaction",
        "ratio_to_median_purchase_price", "repeat_retailer", "used_chip",
        "used_pin_number", "online_order", "is_fraud"
    ]

    rows = []
    for i in range(n):
        is_fraud = 1 if random.random() < fraud_rate else 0

        if is_fraud:
            amount = round(random.uniform(200, 5000), 2)
            distance_home = round(random.uniform(50, 500), 1)
            distance_last = round(random.uniform(30, 300), 1)
            ratio_median = round(random.uniform(2.0, 10.0), 2)
            repeat_retailer = random.choice([0, 0, 0, 1])
            used_chip = random.choice([0, 0, 1])
            used_pin = random.choice([0, 0, 0, 1])
            online_order = random.choice([1, 1, 1, 0])
        else:
            amount = round(random.uniform(5, 500), 2)
            distance_home = round(random.uniform(0, 50), 1)
            distance_last = round(random.uniform(0, 30), 1)
            ratio_median = round(random.uniform(0.3, 2.5), 2)
            repeat_retailer = random.choice([1, 1, 1, 0])
            used_chip = random.choice([1, 1, 0])
            used_pin = random.choice([1, 1, 0])
            online_order = random.choice([0, 0, 1])

        row = [
            f"TXN-{i+1:05d}",
            amount,
            random.choice(["grocery", "restaurant", "gas_station", "online_retail", "travel", "entertainment"]),
            random.randint(0, 23),
            random.randint(0, 6),
            distance_home,
            distance_last,
            ratio_median,
            repeat_retailer,
            used_chip,
            used_pin,
            online_order,
            is_fraud,
        ]
        rows.append(row)

    return header, rows


if __name__ == "__main__":
    header, rows = generate_dataset()
    os.makedirs("data", exist_ok=True)
    with open("data/fraud_detection.csv", "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(header)
        writer.writerows(rows)
    fraud_count = sum(1 for r in rows if r[-1] == 1)
    print(f"Generated {len(rows)} rows ({fraud_count} fraud, {len(rows)-fraud_count} legit)")
