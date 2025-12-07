import secrets
from collections import Counter
import math


def calculate_entropy(data):
    if not data:
        return 0
    length = len(data)
    counts = Counter(data)
    entropy = 0
    for count in counts.values():
        p_x = count / length
        entropy += -p_x * math.log2(p_x)
    return entropy


test_data = {
    "dev_random": open("/dev/random", "rb").read(100000),
    "dev_urandom": open("/dev/urandom", "rb").read(100000),
    "secrets": secrets.token_bytes(100000),
    "pattern": bytes(range(64)) * (100000 // 64) + bytes(range(100000 % 64)),
    "zeros": b"\x00" * 100000,
}


entropies = {}
for source, data in test_data.items():
    entropies[source] = calculate_entropy(data)
    print(f"{source}: Entropy = {entropies[source]:.4f} bits per byte")
