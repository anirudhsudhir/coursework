import pytest
from order_processor import calculate_discount, update_order_status

#Example test cases:
def test_regular_low_amount():
    assert calculate_discount("regular", 500) == 0

def test_premium_discount():
    assert calculate_discount("premium", 2000) == 200

#Write the rest of the test cases so that the code is 90- 100% covered.

# Additional test cases for calculate_discount
def test_regular_high_amount():
    """Test regular customer with amount > 1000"""
    assert calculate_discount("regular", 1500) == 75  # 1500 * 0.05

def test_vip_low_amount():
    """Test VIP customer with amount <= 5000"""
    assert calculate_discount("vip", 3000) == 300  # 3000 * 0.1

def test_vip_high_amount():
    """Test VIP customer with amount > 5000"""
    assert calculate_discount("vip", 6000) == 1200  # 6000 * 0.2

def test_unknown_customer_type():
    """Test that unknown customer type raises ValueError"""
    with pytest.raises(ValueError, match="Unknown customer type"):
        calculate_discount("unknown", 1000)

# Test cases for update_order_status
def test_pending_paid_order():
    """Test pending order that is paid becomes processing"""
    order = {"status": "pending", "paid": True, "items_available": False}
    assert update_order_status(order) == "processing"
    assert order["status"] == "processing"

def test_pending_unpaid_order():
    """Test pending order that is not paid becomes awaiting_payment"""
    order = {"status": "pending", "paid": False, "items_available": False}
    assert update_order_status(order) == "awaiting_payment"
    assert order["status"] == "awaiting_payment"

def test_processing_items_available():
    """Test processing order with items available becomes shipped"""
    order = {"status": "processing", "paid": True, "items_available": True}
    assert update_order_status(order) == "shipped"
    assert order["status"] == "shipped"

def test_processing_items_unavailable():
    """Test processing order with items unavailable becomes backorder"""
    order = {"status": "processing", "paid": True, "items_available": False}
    assert update_order_status(order) == "backorder"
    assert order["status"] == "backorder"

