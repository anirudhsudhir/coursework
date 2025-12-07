def calculate_discount(customer_type, total_amount):
    if customer_type == "regular":
        if total_amount > 1000:
            return total_amount * 0.05
        else:
            return 0
    elif customer_type == "premium":
        return total_amount * 0.1
    elif customer_type == "vip":
        if total_amount > 5000:
            return total_amount * 0.2
        else:
            return total_amount * 0.1
    else:
        raise ValueError("Unknown customer type")


def update_order_status(order):
    if order["status"] == "pending":
        if order["paid"]:
            order["status"] = "processing"
        else:
            order["status"] = "awaiting_payment"
    elif order["status"] == "processing":
        if order["items_available"]:
            order["status"] = "shipped"
        else:
            order["status"] = "backorder"
    return order["status"]