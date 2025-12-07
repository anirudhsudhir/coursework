def text_to_numbers(text):
    return [ord(char) - ord('A') for char in text.upper() if char.isalpha()]

def numbers_to_text(numbers):
    return ''.join([chr(num + ord('A')) for num in numbers])

def pad_text(text, block_size):
    while len(text) % block_size != 0:
        text += 'X'
    return text

def matrix_multiply(a, b, mod):
    rows_a, cols_a = len(a), len(a[0])
    rows_b, cols_b = len(b), len(b[0])
    result = [[0 for _ in range(cols_b)] for _ in range(rows_a)]
    
    for i in range(rows_a):
        for j in range(cols_b):
            for k in range(cols_a):
                result[i][j] += a[i][k] * b[k][j]
            result[i][j] %= mod
    return result

def matrix_vector_multiply(matrix, vector, mod):
    result = [0] * len(matrix)
    for i in range(len(matrix)):
        for j in range(len(vector)):
            result[i] += matrix[i][j] * vector[j]
        result[i] %= mod
    return result

def determinant_2x2(matrix):
    return matrix[0][0] * matrix[1][1] - matrix[0][1] * matrix[1][0]

def determinant_3x3(matrix):
    return (matrix[0][0] * (matrix[1][1] * matrix[2][2] - matrix[1][2] * matrix[2][1]) -
            matrix[0][1] * (matrix[1][0] * matrix[2][2] - matrix[1][2] * matrix[2][0]) +
            matrix[0][2] * (matrix[1][0] * matrix[2][1] - matrix[1][1] * matrix[2][0]))

def matrix_determinant(matrix):
    size = len(matrix)
    if size == 2:
        return determinant_2x2(matrix)
    elif size == 3:
        return determinant_3x3(matrix)

def extended_gcd(a, b):
    if a == 0:
        return b, 0, 1
    gcd, x1, y1 = extended_gcd(b % a, a)
    x = y1 - (b // a) * x1
    y = x1
    return gcd, x, y

def mod_inverse(a, m):
    gcd, x, y = extended_gcd(a, m)
    if gcd != 1:
        return None
    return (x % m + m) % m

def matrix_inverse_2x2(matrix, mod):
    det = determinant_2x2(matrix) % mod
    det_inv = mod_inverse(det, mod)
    if det_inv is None:
        return None
    
    inv_matrix = [[matrix[1][1], -matrix[0][1]],
                  [-matrix[1][0], matrix[0][0]]]
    
    for i in range(2):
        for j in range(2):
            inv_matrix[i][j] = (inv_matrix[i][j] * det_inv) % mod
    
    return inv_matrix

def matrix_inverse_3x3(matrix, mod):
    det = determinant_3x3(matrix) % mod
    det_inv = mod_inverse(det, mod)
    if det_inv is None:
        return None
    
    adj_matrix = [
        [(matrix[1][1] * matrix[2][2] - matrix[1][2] * matrix[2][1]),
         -(matrix[0][1] * matrix[2][2] - matrix[0][2] * matrix[2][1]),
         (matrix[0][1] * matrix[1][2] - matrix[0][2] * matrix[1][1])],
        [-(matrix[1][0] * matrix[2][2] - matrix[1][2] * matrix[2][0]),
         (matrix[0][0] * matrix[2][2] - matrix[0][2] * matrix[2][0]),
         -(matrix[0][0] * matrix[1][2] - matrix[0][2] * matrix[1][0])],
        [(matrix[1][0] * matrix[2][1] - matrix[1][1] * matrix[2][0]),
         -(matrix[0][0] * matrix[2][1] - matrix[0][1] * matrix[2][0]),
         (matrix[0][0] * matrix[1][1] - matrix[0][1] * matrix[1][0])]
    ]
    
    for i in range(3):
        for j in range(3):
            adj_matrix[i][j] = (adj_matrix[i][j] * det_inv) % mod
    
    return adj_matrix

def matrix_inverse(matrix, mod):
    size = len(matrix)
    if size == 2:
        return matrix_inverse_2x2(matrix, mod)
    elif size == 3:
        return matrix_inverse_3x3(matrix, mod)

def encrypt(plaintext, key_matrix):
    n = len(key_matrix)
    plaintext = pad_text(plaintext.upper().replace(' ', ''), n)
    numbers = text_to_numbers(plaintext)
    
    encrypted_numbers = []
    for i in range(0, len(numbers), n):
        block = numbers[i:i+n]
        encrypted_block = matrix_vector_multiply(key_matrix, block, 26)
        encrypted_numbers.extend(encrypted_block)
    
    return numbers_to_text(encrypted_numbers)

def create_key_matrix(key_string, size):
    numbers = text_to_numbers(key_string)
    if len(numbers) < size * size:
        numbers.extend([0] * (size * size - len(numbers)))
    
    matrix = []
    for i in range(size):
        row = numbers[i*size:(i+1)*size]
        matrix.append(row)
    return matrix

def print_matrix(matrix):
    for row in matrix:
        print(row)

pt = input("Enter the plaintext: ")
key_size = int(input("Enter the key size: "))
key_matrix = input("Enter the key: ")
numbers = list(map(int, key_matrix.split()))
   
key = []
for i in range(key_size):
    row = numbers[i*key_size:(i+1)*key_size]
    key.append(row)

ct = encrypt(pt, key)
print("Ciphertext is: ", ct)