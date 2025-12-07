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


if __name__ == "__main__":
    oracle = PaddingOracle('10.9.0.80', 5000)

    # Get the IV + Ciphertext from the oracle
    iv_and_ctext = bytearray(oracle.ctext)
    IV    = iv_and_ctext[00:16]
    C1    = iv_and_ctext[16:32]  # 1st block of ciphertext
    C2    = iv_and_ctext[32:48]  # 2nd block of ciphertext
    print("C1:  " + C1.hex())
    print("C2:  " + C2.hex())

    ###############################################################
    # Here, we initialize D2 with C1, so when they are XOR-ed,
    # The result is 0. This is not required for the attack.
    # Its sole purpose is to make the printout look neat.
    # In the experiment, we will iteratively replace these values.
    D2 = bytearray(16)

    D2[0]  = 0xb8
    D2[1]  = 0x90
    D2[2]  = 0x66
    D2[3]  = 0xf
    D2[4]  = 0x5c
    D2[5]  = 0x22
    D2[6]  = 0x66
    D2[7]  = 0x08
    D2[8]  = 0xcb
    D2[9]  = 0x9a
    D2[10] = 0xec
    D2[11] = 0x45
    D2[12] = 0x1c
    D2[13] = 0xf1
    D2[14] = 0x3b
    D2[15] = 0xce
    ###############################################################
    # In the experiment, we need to iteratively modify CC1
    # We will send this CC1 to the oracle, and see its response.
    CC1 = bytearray(16)

    K = 17

    CC1[0]  = D2[0]^K
    CC1[1]  = D2[1]^K
    CC1[2]  = D2[2]^K
    CC1[3]  = D2[3]^K
    CC1[4]  = D2[4]^K
    CC1[5]  = D2[5]^K
    CC1[6]  = D2[6]^K
    CC1[7]  = D2[7]^K
    CC1[8]  = D2[8]^K
    CC1[9]  = D2[9]^K
    CC1[10] = D2[10]^K
    CC1[11] = D2[11]^K
    CC1[12] = D2[12]^K
    CC1[13] = D2[13]^K
    CC1[14] = D2[14]^K
    CC1[15] = D2[15]^K

    ###############################################################
    # In each iteration, we focus on one byte of CC1.  
    # We will try all 256 possible values, and send the constructed
    # ciphertext CC1 + C2 (plus the IV) to the oracle, and see 
    # which value makes the padding valid. 
    # As long as our construction is correct, there will be 
    # one valid value. This value helps us get one byte of D2. 
    # Repeating the method for 16 times, we get all the 16 bytes of D2.
    for i in range(256):
          CC1[16 - K] = i
          status = oracle.decrypt(IV + CC1 + C2)
          if status == "Valid":
              print("Valid: i = 0x{:02x}".format(i))
              print("CC1: " + CC1.hex())
    ###############################################################

    # Once you get all the 16 bytes of D2, you can easily get P2
    P2 = xor(C1, D2)
    print("P2:  " + P2.hex())
