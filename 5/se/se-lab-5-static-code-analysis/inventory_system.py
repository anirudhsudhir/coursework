"""Module for managing inventory system with stock operations"""
import json
from datetime import datetime

# Global variable
stock_data = {}


def add_item(item="default", qty=0, logs=None):
    """add items to inventory with quantity tracking"""
    if logs is None:
        logs = []
    if not item:
        return
    stock_data[item] = stock_data.get(item, 0) + qty
    logs.append(f"{datetime.now()}: Added {qty} of {item}")


def remove_item(item, qty):
    """remove items from inventory by quantity"""
    try:
        stock_data[item] -= qty
        if stock_data[item] <= 0:
            del stock_data[item]
    except KeyError:
        pass


def get_qty(item):
    """get quantity of an item in inventory"""
    return stock_data[item]


def load_data(file="inventory.json"):
    """load inventory data from JSON file"""
    global stock_data
    with open(file, "r", encoding="utf-8") as f:
        stock_data = json.loads(f.read())


def save_data(file="inventory.json"):
    """save inventory data to JSON file"""
    with open(file, "w", encoding="utf-8") as f:
        f.write(json.dumps(stock_data))


def print_data():
    """print current inventory items and quantities"""
    print("Items Report")
    for i in stock_data:
        print(i, "->", stock_data[i])


def check_low_items(threshold=5):
    """check and return items below the threshold quantity"""
    result = []
    for i in stock_data:
        if stock_data[i] < threshold:
            result.append(i)
    return result


def main():
    """main function to demonstrate inventory operations"""
    add_item("apple", 10)
    add_item("banana", -2)
    add_item(123, "ten")  # invalid types, no check
    remove_item("apple", 3)
    remove_item("orange", 1)
    print("Apple stock:", get_qty("apple"))
    print("Low items:", check_low_items())
    save_data()
    load_data()
    print_data()
    print('eval used')


main()
