def encrypt(pt: str, key: int) -> str:
    return "".join([chr(((ord(ch) - 65) + key) % 26 + 65) if ch.isalpha() else ch for ch in pt])
def decrypt(ct: str, key: int) -> str:
    return "".join([chr(((ord(ch) - 65) - key) % 26 + 65) if ch.isalpha() else ch for ch in ct])

pt = input("Enter the plaintext: ")
key = int(input("Enter the key for Caesar cipher: "))

ct = encrypt(pt, key)
print("Ciphertext is: ", ct)
print("Decrypted Piphertext is: ", decrypt(ct, key))