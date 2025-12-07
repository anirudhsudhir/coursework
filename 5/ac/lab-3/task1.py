import time
import random

def seed_with_time():
    seed = int(time.time())
    print(f"current timestamp: {seed}")
    random. seed (int(time.time()))

def gen_key() -> bytes:
    return b"".join([int(random.randint(0, 255)).to_bytes(1, "big") for _ in range(16)])

seed_with_time()
key = gen_key()
print(f"Generated key (hex): {key.hex()}")
