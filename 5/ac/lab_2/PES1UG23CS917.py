import random
import string
from functools import reduce

with open('PES1UG23CS917.txt') as f:

    # q1
    print("\nQuestion 1")
    content = f.read().lower()
    print("Printing the file contents: ", content)

    # q2
    print("\nQuestion 2")
    words_count = ['th', 'he', 'ar', 'ing', 'e', 'or']
    print("Word Frequency")
    for word in words_count:
        wc = content.count(word)
        print(word, " - ", wc)

    alphabets = string.ascii_lowercase

    # q3
    print("\nQuestion 3")
    key = list(alphabets)
    random.shuffle(key)
    print("Printing the key")
    for idx in range(len(key)):
        print(f"{alphabets[idx]} -> {key[idx]}", end=", ")
    print()

    # q4
    print("\nQuestion 4")
    ciphertext = ''.join([key[ord(letter) - 97] if letter.isalpha() else letter for letter in content])
    print("Ciphertext is: \n", ciphertext)

    # q5
    print("\nQuestion 5")
    plaintext = ''.join([alphabets[key.index(letter)] if letter.isalpha() else letter for letter in ciphertext])
    print("Plaintext is: \n", plaintext)

    #q7
    # print("\nQuestion 7: Ciphertext letter frequency")
    print("\n")
    q7_cipher = "BFPXTRWTOW CEN BRTEWTA JNFVU E NJLNWFWJWFGV BFPXTR CXTRT TEBX QTWWTR MEPN WG EVGWXTR VGWFBT WXT PEWWTRV GZ WXT CGRAN WXT BGMMGV QTWWTRN GZ WXT TVUQFNX EQPXELTW XTQP FV ATBGAFVU E NMEQQ UJTNN EW ZFRNW WTNWFVU WXTM FV WXT WTOW EVA WXT RTNJQW LTBGMTN BQTERTR WXT PRGBTNN RTDTEQN WXT NTBRTW GZ WXT BGAT GVBT WXT MEPPFVU FN BGRRTBW WXT MTNNEUT XFAATV FN WXEW NJLNWFWJWFGV BFPXTRN ERT NFMPQT LJW ZRTIJTVBK EVEQKNFN MESTN WXTM CTES"
    def accumulate(acc, pair):
        word, count = pair
        acc[word] = acc.get(word, 0) + count
        return acc
    print(reduce(accumulate, list(map(lambda letter: (letter, 1), q7_cipher)), {}))