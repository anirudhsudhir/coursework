#!/usr/bin/python3
import socket
from binascii import hexlify, unhexlify

# XOR two bytearrays
def xor(first, second):
    return bytearray(x^y for x,y in zip(first, second))

class PaddingOracle:
    def __init__(self, host, port) -> None:
        self.s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.s.connect((host, port))
        ciphertext = self.s.recv(4096).decode().strip()
        self.ctext = unhexlify(ciphertext)
    
    def decrypt(self, ctext: bytes) -> None:
        self._send(hexlify(ctext))
        return self._recv()
    
    def _recv(self):
        resp = self.s.recv(4096).decode().strip()
        return resp
    
    def _send(self, hexstr: bytes):
        self.s.send(hexstr + b'\n')
    
    def __del__(self):
        self.s.close()

def decrypt_block(oracle, IV, C_prev, C_target):
    """Decrypt a single block using padding oracle attack"""
    D = bytearray(16)
    CC = bytearray(16)
    
    for K in range(1, 17):
        
        for i in range(256):
            CC[16 - K] = i
            status = oracle.decrypt(IV + CC + C_target)
            
            if status == "Valid":
                # Calculate D[16-K] = i XOR K
                D[16 - K] = i ^ K
                # Update all previous bytes for next padding value (K+1)
                for j in range(K):
                    CC[16 - K + j] = D[16 - K + j] ^ (K + 1)
                
                break
    
    # Decrypt plaintext: P = C_prev XOR D
    P = xor(C_prev, D)
    return D, P

if __name__ == "__main__":
    oracle = PaddingOracle('10.9.0.80', 6000)
    
    # Get the IV + Ciphertext from the oracle
    iv_and_ctext = bytearray(oracle.ctext)
    print(f"Total length: {len(iv_and_ctext)} bytes")
    
    IV = iv_and_ctext[00:16]
    C1 = iv_and_ctext[16:32]
    C2 = iv_and_ctext[32:48]
    C3 = iv_and_ctext[48:64]
    
    print("IV: " + IV.hex())
    print("C1: " + C1.hex())
    print("C2: " + C2.hex())
    print("C3: " + C3.hex())

    D1, P1 = decrypt_block(oracle, IV, IV, C1)
    D2, P2 = decrypt_block(oracle, IV, C1, C2)
    D3, P3 = decrypt_block(oracle, IV, C2, C3)

    # Complete plaintext
    print("\n" + "="*60)
    print("COMPLETE PLAINTEXT")
    print("="*60)
    complete_plaintext = P1 + P2 + P3
    print("Full plaintext: " + complete_plaintext.hex())
    print("Full plaintext (ASCII): " + complete_plaintext.decode('ascii', errors='replace'))
