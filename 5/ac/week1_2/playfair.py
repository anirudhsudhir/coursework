import string

def create_grid(key):
    key = key.upper().replace('J', 'I')
    
    seen = set()
    key_chars = []
    for char in key:
        if char.isalpha() and char not in seen:
            key_chars.append(char)
            seen.add(char)
    
    for char in string.ascii_uppercase:
        if char not in seen:
            key_chars.append(char)
    
    grid = []
    for i in range(5):
        row = key_chars[i*5:(i+1)*5]
        grid.append(row)
    
    return grid

def create_position_dict(grid):
    positions = {}
    for row in range(5):
        for col in range(5):
            positions[grid[row][col]] = (row, col)
    return positions

def prepare_text(text):
    text = text.upper().replace('J', 'I')
    clean_text = ''.join(char for char in text if char.isalpha())
    
    pairs = []
    i = 0
    while i < len(clean_text):
        if i == len(clean_text) - 1:
            pairs.append(clean_text[i] + 'X')
            break
        elif clean_text[i] == clean_text[i + 1]:
            pairs.append(clean_text[i] + 'X')
            i += 1
        else:
            pairs.append(clean_text[i:i+2])
            i += 2
    
    return pairs

def encrypt_pair(pair, grid, positions):
    char1, char2 = pair[0], pair[1]
    row1, col1 = positions[char1]
    row2, col2 = positions[char2]
    
    if row1 == row2:
        new_col1 = (col1 + 1) % 5
        new_col2 = (col2 + 1) % 5
        return grid[row1][new_col1] + grid[row2][new_col2]
    elif col1 == col2:
        new_row1 = (row1 + 1) % 5
        new_row2 = (row2 + 1) % 5
        return grid[new_row1][col1] + grid[new_row2][col2]
    else:
        return grid[row1][col2] + grid[row2][col1]

def decrypt_pair(pair, grid, positions):
    char1, char2 = pair[0], pair[1]
    row1, col1 = positions[char1]
    row2, col2 = positions[char2]
    
    if row1 == row2:
        new_col1 = (col1 - 1) % 5
        new_col2 = (col2 - 1) % 5
        return grid[row1][new_col1] + grid[row2][new_col2]
    elif col1 == col2:
        new_row1 = (row1 - 1) % 5
        new_row2 = (row2 - 1) % 5
        return grid[new_row1][col1] + grid[new_row2][col2]
    else:
        return grid[row1][col2] + grid[row2][col1]

def encrypt(plaintext, key):
    grid = create_grid(key)
    positions = create_position_dict(grid)
    pairs = prepare_text(plaintext)
    
    encrypted_text = ''
    for pair in pairs:
        encrypted_text += encrypt_pair(pair, grid, positions)
    
    return encrypted_text

def decrypt(ciphertext, key):
    grid = create_grid(key)
    positions = create_position_dict(grid)
    
    pairs = [ciphertext[i:i+2] for i in range(0, len(ciphertext), 2)]
    
    decrypted_text = ''
    for pair in pairs:
        decrypted_text += decrypt_pair(pair, grid, positions)
    
    return decrypted_text


pt = input("Enter the plaintext: ")
key = input("Enter the key: ")

ct = encrypt(pt, key)
print("Ciphertext is: ", ct)
print("Decrypted Piphertext is: ", decrypt(ct, key))