import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
import tensorflow as tf
from tensorflow.keras import layers, models
import time
import urllib.request
from pathlib import Path

CLASSES = ['cat', 'car', 'apple']
SAMPLES_PER_CLASS = 2000
IMG_SIZE = 28
OUTPUT_DIR = Path('output')

OUTPUT_DIR.mkdir(exist_ok=True)

print("Quick Doodle Classifier - Fast Test Version")
print("="*50)

data_dir = Path('data')
data_dir.mkdir(exist_ok=True)

X_all = []
y_all = []

for idx, class_name in enumerate(CLASSES):
    file_path = data_dir / f'{class_name}.npy'
    
    if not file_path.exists():
        print(f"Downloading {class_name}...")
        url = f'https://storage.googleapis.com/quickdraw_dataset/full/numpy_bitmap/{class_name}.npy'
        urllib.request.urlretrieve(url, file_path)
    
    data = np.load(file_path)[:SAMPLES_PER_CLASS]
    X_all.append(data)
    y_all.extend([idx] * len(data))

X = np.concatenate(X_all, axis=0).astype('float32')
y = np.array(y_all)

indices = np.random.permutation(len(X))
X = X[indices]
y = y[indices]

X = (X > 127).astype('float32')

n = len(X)
train_end = int(n * 0.7)
val_end = int(n * 0.85)

X_train, y_train = X[:train_end], y[:train_end]
X_val, y_val = X[train_end:val_end], y[train_end:val_end]
X_test, y_test = X[val_end:], y[val_end:]

print(f"\nDataset: {len(X)} images, {len(CLASSES)} classes")
print(f"Train: {len(X_train)}, Val: {len(X_val)}, Test: {len(X_test)}")

from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train.reshape(len(X_train), -1))
X_val_scaled = scaler.transform(X_val.reshape(len(X_val), -1))

print("\n1. Training Logistic Regression...")
lr = LogisticRegression(max_iter=200, solver='lbfgs', random_state=42)
start = time.time()
lr.fit(X_train_scaled, y_train)
lr_time = time.time() - start
lr_acc = accuracy_score(y_val, lr.predict(X_val_scaled))
print(f"   Accuracy: {lr_acc*100:.2f}% | Time: {lr_time:.2f}s")

print("\n2. Training Simple CNN...")
model = models.Sequential([
    layers.Input(shape=(IMG_SIZE, IMG_SIZE, 1)),
    layers.Conv2D(32, 3, activation='relu'),
    layers.MaxPooling2D(2),
    layers.Flatten(),
    layers.Dense(64, activation='relu'),
    layers.Dropout(0.25),
    layers.Dense(len(CLASSES), activation='softmax')
])

model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

start = time.time()
history = model.fit(
    X_train.reshape(-1, IMG_SIZE, IMG_SIZE, 1), y_train,
    validation_data=(X_val.reshape(-1, IMG_SIZE, IMG_SIZE, 1), y_val),
    epochs=5,
    batch_size=128,
    verbose=0
)
cnn_time = time.time() - start
cnn_acc = history.history['val_accuracy'][-1]
print(f"   Accuracy: {cnn_acc*100:.2f}% | Time: {cnn_time:.2f}s")

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))

models_names = ['Logistic\nRegression', 'Simple\nCNN']
accuracies = [lr_acc * 100, cnn_acc * 100]
times = [lr_time, cnn_time]

ax1.bar(models_names, accuracies, color=['#3498db', '#e74c3c'])
ax1.set_ylabel('Accuracy (%)')
ax1.set_title('Model Accuracy')
ax1.set_ylim([0, 100])
ax1.grid(True, alpha=0.3, axis='y')
for i, v in enumerate(accuracies):
    ax1.text(i, v + 2, f'{v:.1f}%', ha='center', fontweight='bold')

ax2.bar(models_names, times, color=['#3498db', '#e74c3c'])
ax2.set_ylabel('Training Time (seconds)')
ax2.set_title('Training Time')
ax2.grid(True, alpha=0.3, axis='y')
for i, v in enumerate(times):
    ax2.text(i, v + max(times)*0.02, f'{v:.1f}s', ha='center', fontweight='bold')

plt.tight_layout()
output_path = OUTPUT_DIR / 'quick_results.png'
plt.savefig(output_path, dpi=150, bbox_inches='tight')
print(f"\nResults saved to {output_path}")

print("\n" + "="*50)
print("Quick test complete!")
print(f"All outputs saved to: {OUTPUT_DIR}/")
print("For full version with all models, run: python main.py")