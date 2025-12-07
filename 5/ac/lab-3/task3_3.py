import time
import secrets

def secrets_generated():
    start = time.time()

    keys = [secrets.token_bytes(128).hex() for _ in range(20)]
    for i, key in enumerate(keys, start=1):
        print(f"Key {i}: {key}")

    end = time.time()
    print(f"Total execution time: {end - start} seconds")


secrets_generated()
