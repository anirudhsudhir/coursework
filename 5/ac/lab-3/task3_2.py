import time

def urandom_generated():
    start = time.time()

    with open("/dev/urandom", "rb") as file:
        keys = [file.read(128).hex() for _ in range(20)]
        for i, key in enumerate(keys, start=1):
            print(f"Key {i}: {key}")

    end = time.time()
    print(f"Total execution time: {end - start} seconds")


urandom_generated()
