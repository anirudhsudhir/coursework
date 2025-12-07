import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, models
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
import time
import os
from pathlib import Path
import warnings

warnings.filterwarnings('ignore', category=FutureWarning)
warnings.filterwarnings('ignore', category=UserWarning)

CLASSES = ['cat', 'tree', 'car', 'apple', 'fish']
SAMPLES_PER_CLASS = 5000
IMG_SIZE = 28
RANDOM_SEED = 42
OUTPUT_DIR = Path('output')

np.random.seed(RANDOM_SEED)
tf.random.set_seed(RANDOM_SEED)

OUTPUT_DIR.mkdir(exist_ok=True)

class DoodleDataset:
    def __init__(self, data_dir='data'):
        self.data_dir = Path(data_dir)
        self.classes = CLASSES
        self.num_classes = len(CLASSES)
        
    def load_data(self):
        print("Loading data...")
        X_data = []
        y_data = []
        
        for idx, class_name in enumerate(self.classes):
            file_path = self.data_dir / f'{class_name}.npy'
            
            if not file_path.exists():
                print(f"Downloading {class_name}...")
                self._download_class(class_name)
            
            data = np.load(file_path)
            data = data[:SAMPLES_PER_CLASS]
            
            X_data.append(data)
            y_data.extend([idx] * len(data))
        
        X_data = np.concatenate(X_data, axis=0)
        y_data = np.array(y_data)
        
        indices = np.random.permutation(len(X_data))
        X_data = X_data[indices]
        y_data = y_data[indices]
        
        return X_data, y_data
    
    def _download_class(self, class_name):
        import urllib.request
        
        self.data_dir.mkdir(parents=True, exist_ok=True)
        
        base_url = 'https://storage.googleapis.com/quickdraw_dataset/full/numpy_bitmap'
        url = f'{base_url}/{class_name}.npy'
        file_path = self.data_dir / f'{class_name}.npy'
        
        urllib.request.urlretrieve(url, file_path)
        print(f"Downloaded {class_name}.npy")
    
    def split_data(self, X, y, train_ratio=0.7, val_ratio=0.15):
        n = len(X)
        train_end = int(n * train_ratio)
        val_end = int(n * (train_ratio + val_ratio))
        
        X_train = X[:train_end]
        y_train = y[:train_end]
        
        X_val = X[train_end:val_end]
        y_val = y[train_end:val_end]
        
        X_test = X[val_end:]
        y_test = y[val_end:]
        
        return (X_train, y_train), (X_val, y_val), (X_test, y_test)
    
    def preprocess_data(self, X, binarize=True):
        X = X.astype('float32')
        
        if binarize:
            X = (X > 127).astype('float32')
        else:
            X = X / 255.0
        
        return X
    
    def visualize_samples(self, X, y, n_samples=10):
        fig, axes = plt.subplots(2, 5, figsize=(15, 6))
        axes = axes.ravel()
        
        for i in range(min(n_samples, len(X))):
            axes[i].imshow(X[i].reshape(IMG_SIZE, IMG_SIZE), cmap='gray')
            axes[i].set_title(f'{self.classes[y[i]]}')
            axes[i].axis('off')
        
        plt.tight_layout()
        output_path = OUTPUT_DIR / 'sample_doodles.png'
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        plt.close()
        print(f"Saved sample doodles to {output_path}")


class SVMModel:
    def __init__(self, num_classes, kernel='rbf'):
        self.num_classes = num_classes
        self.kernel = kernel
        
        if kernel == 'linear':
            self.model = SVC(kernel='linear', random_state=RANDOM_SEED, max_iter=500)
        elif kernel == 'rbf':
            self.model = SVC(kernel='rbf', gamma='scale', random_state=RANDOM_SEED, max_iter=500)
        elif kernel == 'poly':
            self.model = SVC(kernel='poly', degree=3, random_state=RANDOM_SEED, max_iter=500)
        
        self.scaler = StandardScaler()
    
    def train(self, X_train, y_train):
        X_flat = X_train.reshape(X_train.shape[0], -1)
        X_scaled = self.scaler.fit_transform(X_flat)
        
        print(f"\nTraining SVM with {self.kernel} kernel...")
        start_time = time.time()
        self.model.fit(X_scaled, y_train)
        training_time = time.time() - start_time
        
        print(f"Training completed in {training_time:.2f} seconds")
        return training_time
    
    def evaluate(self, X, y):
        X_flat = X.reshape(X.shape[0], -1)
        X_scaled = self.scaler.transform(X_flat)
        y_pred = self.model.predict(X_scaled)
        accuracy = accuracy_score(y, y_pred)
        return accuracy, y_pred


class LogisticRegressionModel:
    def __init__(self, num_classes):
        self.num_classes = num_classes
        self.model = LogisticRegression(
            max_iter=200,
            solver='lbfgs',
            random_state=RANDOM_SEED,
            n_jobs=-1
        )
        self.scaler = StandardScaler()
    
    def train(self, X_train, y_train):
        X_flat = X_train.reshape(X_train.shape[0], -1)
        X_scaled = self.scaler.fit_transform(X_flat)
        
        print("\nTraining Logistic Regression...")
        start_time = time.time()
        self.model.fit(X_scaled, y_train)
        training_time = time.time() - start_time
        
        print(f"Training completed in {training_time:.2f} seconds")
        return training_time
    
    def evaluate(self, X, y):
        X_flat = X.reshape(X.shape[0], -1)
        X_scaled = self.scaler.transform(X_flat)
        y_pred = self.model.predict(X_scaled)
        accuracy = accuracy_score(y, y_pred)
        return accuracy, y_pred


class SimpleCNN:
    def __init__(self, num_classes, version='v1'):
        self.num_classes = num_classes
        self.version = version
        self.model = self._build_model()
        self.history = None
    
    def _build_model(self):
        model = models.Sequential()
        
        if self.version == 'v1':
            model.add(layers.Input(shape=(IMG_SIZE, IMG_SIZE, 1)))
            model.add(layers.Conv2D(32, (3, 3), activation='relu'))
            model.add(layers.MaxPooling2D((2, 2)))
            model.add(layers.Conv2D(64, (3, 3), activation='relu'))
            model.add(layers.MaxPooling2D((2, 2)))
            model.add(layers.Flatten())
            model.add(layers.Dense(128, activation='relu'))
            model.add(layers.Dropout(0.25))
            model.add(layers.Dense(self.num_classes, activation='softmax'))
        
        elif self.version == 'v2':
            model.add(layers.Input(shape=(IMG_SIZE, IMG_SIZE, 1)))
            model.add(layers.Conv2D(32, (3, 3), activation='relu'))
            model.add(layers.MaxPooling2D((2, 2)))
            model.add(layers.Flatten())
            model.add(layers.Dense(128, activation='relu'))
            model.add(layers.Dropout(0.25))
            model.add(layers.Dense(self.num_classes, activation='softmax'))
        
        elif self.version == 'v3':
            model.add(layers.Input(shape=(IMG_SIZE, IMG_SIZE, 1)))
            model.add(layers.Conv2D(32, (3, 3), activation='relu'))
            model.add(layers.MaxPooling2D((2, 2)))
            model.add(layers.Flatten())
            model.add(layers.Dense(64, activation='relu'))
            model.add(layers.Dropout(0.25))
            model.add(layers.Dense(self.num_classes, activation='softmax'))
        
        model.compile(
            optimizer='adam',
            loss='sparse_categorical_crossentropy',
            metrics=['accuracy']
        )
        
        return model
    
    def train(self, X_train, y_train, X_val, y_val, epochs=10, batch_size=128):
        X_train_reshaped = X_train.reshape(-1, IMG_SIZE, IMG_SIZE, 1)
        X_val_reshaped = X_val.reshape(-1, IMG_SIZE, IMG_SIZE, 1)
        
        callbacks = [
            EarlyStopping(
                monitor='val_loss',
                patience=5,
                restore_best_weights=True,
                verbose=0
            ),
            ReduceLROnPlateau(
                monitor='val_loss',
                factor=0.5,
                patience=3,
                min_lr=1e-6,
                verbose=0
            )
        ]
        
        print(f"\nTraining CNN {self.version}...")
        start_time = time.time()
        
        self.history = self.model.fit(
            X_train_reshaped, y_train,
            validation_data=(X_val_reshaped, y_val),
            epochs=epochs,
            batch_size=batch_size,
            callbacks=callbacks,
            verbose=1
        )
        
        training_time = time.time() - start_time
        print(f"Training completed in {training_time:.2f} seconds")
        
        return training_time
    
    def evaluate(self, X, y):
        X_reshaped = X.reshape(-1, IMG_SIZE, IMG_SIZE, 1)
        loss, accuracy = self.model.evaluate(X_reshaped, y, verbose=0)
        y_pred = np.argmax(self.model.predict(X_reshaped, verbose=0), axis=1)
        return accuracy, y_pred
    
    def plot_history(self, filename):
        if self.history is None:
            return
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
        
        ax1.plot(self.history.history['accuracy'], label='Train Accuracy')
        ax1.plot(self.history.history['val_accuracy'], label='Val Accuracy')
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('Accuracy')
        ax1.set_title(f'CNN {self.version} - Accuracy')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        ax2.plot(self.history.history['loss'], label='Train Loss')
        ax2.plot(self.history.history['val_loss'], label='Val Loss')
        ax2.set_xlabel('Epoch')
        ax2.set_ylabel('Loss')
        ax2.set_title(f'CNN {self.version} - Loss')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        output_path = OUTPUT_DIR / filename
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        plt.close()
        print(f"Saved training history to {output_path}")


class Evaluator:
    def __init__(self, class_names):
        self.class_names = class_names
    
    def plot_confusion_matrix(self, y_true, y_pred, title, filename):
        cm = confusion_matrix(y_true, y_pred)
        cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm_normalized, annot=True, fmt='.2f', cmap='Blues',
                    xticklabels=self.class_names,
                    yticklabels=self.class_names)
        plt.title(title)
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        plt.tight_layout()
        output_path = OUTPUT_DIR / filename
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        plt.close()
        print(f"Saved confusion matrix to {output_path}")
    
    def print_classification_report(self, y_true, y_pred, model_name):
        print(f"\n{model_name} Classification Report:")
        print("="*60)
        report = classification_report(y_true, y_pred, target_names=self.class_names)
        print(report)
    
    def plot_comparison(self, results, filename):
        models = list(results.keys())
        accuracies = [results[m]['accuracy'] * 100 for m in models]
        times = [results[m]['training_time'] for m in models]
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
        
        colors = plt.cm.Set3(np.linspace(0, 1, len(models)))
        
        bars1 = ax1.bar(range(len(models)), accuracies, color=colors)
        ax1.set_xticks(range(len(models)))
        ax1.set_xticklabels(models, rotation=45, ha='right')
        ax1.set_ylabel('Accuracy (%)')
        ax1.set_title('Model Accuracy Comparison')
        ax1.set_ylim([0, 100])
        ax1.grid(True, alpha=0.3, axis='y')
        for i, (v, bar) in enumerate(zip(accuracies, bars1)):
            ax1.text(bar.get_x() + bar.get_width()/2, v + 1, f'{v:.1f}%', 
                    ha='center', fontweight='bold', fontsize=9)
        
        bars2 = ax2.bar(range(len(models)), times, color=colors)
        ax2.set_xticks(range(len(models)))
        ax2.set_xticklabels(models, rotation=45, ha='right')
        ax2.set_ylabel('Training Time (seconds)')
        ax2.set_title('Training Time Comparison')
        ax2.grid(True, alpha=0.3, axis='y')
        for i, (v, bar) in enumerate(zip(times, bars2)):
            ax2.text(bar.get_x() + bar.get_width()/2, v + max(times)*0.02, 
                    f'{v:.1f}s', ha='center', fontweight='bold', fontsize=9)
        
        plt.tight_layout()
        output_path = OUTPUT_DIR / filename
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        plt.close()
        print(f"Saved comparison plot to {output_path}")
    
    def plot_misclassified_samples(self, X, y_true, y_pred, filename, n_samples=10):
        incorrect_indices = np.where(y_pred != y_true)[0]
        
        if len(incorrect_indices) == 0:
            print("No misclassified samples found!")
            return
        
        n_samples = min(n_samples, len(incorrect_indices))
        sample_indices = np.random.choice(incorrect_indices, n_samples, replace=False)
        
        fig, axes = plt.subplots(2, 5, figsize=(15, 6))
        axes = axes.ravel()
        
        for i, idx in enumerate(sample_indices):
            axes[i].imshow(X[idx].reshape(IMG_SIZE, IMG_SIZE), cmap='gray')
            axes[i].set_title(f'True: {self.class_names[y_true[idx]]}\nPred: {self.class_names[y_pred[idx]]}',
                            fontsize=9)
            axes[i].axis('off')
        
        plt.tight_layout()
        output_path = OUTPUT_DIR / filename
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        plt.close()
        print(f"Saved misclassified samples to {output_path}")
    
    def generate_summary_report(self, results, test_results, filename='performance_summary.txt'):
        output_path = OUTPUT_DIR / filename
        with open(output_path, 'w') as f:
            f.write("="*70 + "\n")
            f.write("QUICK DOODLE CLASSIFIER - PERFORMANCE SUMMARY\n")
            f.write("="*70 + "\n\n")
            
            f.write("DATASET INFORMATION:\n")
            f.write(f"Classes: {', '.join(self.class_names)}\n")
            f.write(f"Samples per class: {SAMPLES_PER_CLASS}\n")
            f.write(f"Total samples: {SAMPLES_PER_CLASS * len(self.class_names)}\n")
            f.write(f"Image size: {IMG_SIZE}x{IMG_SIZE}\n\n")
            
            f.write("VALIDATION SET RESULTS:\n")
            f.write("-"*70 + "\n")
            f.write(f"{'Model':<25} {'Accuracy':>12} {'Time (s)':>12} {'Parameters':>15}\n")
            f.write("-"*70 + "\n")
            
            param_counts = {
                'Logistic Regression': IMG_SIZE * IMG_SIZE * len(self.class_names),
                'SVM (Linear)': 'N/A',
                'SVM (Rbf)': 'N/A',
                'SVM (Poly)': 'N/A',
                'CNN v1': '~180K',
                'CNN v2': '~130K',
                'CNN v3': '~65K'
            }
            
            for model_name, metrics in results.items():
                params = param_counts.get(model_name, 'N/A')
                f.write(f"{model_name:<25} {metrics['accuracy']*100:>11.2f}% {metrics['training_time']:>11.2f}s {str(params):>15}\n")
            
            f.write("\n" + "="*70 + "\n")
            f.write("TEST SET RESULTS (FINAL EVALUATION):\n")
            f.write("-"*70 + "\n")
            f.write(f"{'Model':<25} {'Test Accuracy':>15}\n")
            f.write("-"*70 + "\n")
            
            for model_name, test_acc in test_results.items():
                f.write(f"{model_name:<25} {test_acc*100:>14.2f}%\n")
            
            best_val_model = max(results.items(), key=lambda x: x[1]['accuracy'])
            best_test_model = max(test_results.items(), key=lambda x: x[1])
            fastest_model = min(results.items(), key=lambda x: x[1]['training_time'])
            
            f.write("\n" + "="*70 + "\n")
            f.write("KEY FINDINGS:\n")
            f.write("-"*70 + "\n")
            f.write(f"Best Validation Accuracy: {best_val_model[0]} ({best_val_model[1]['accuracy']*100:.2f}%)\n")
            f.write(f"Best Test Accuracy: {best_test_model[0]} ({best_test_model[1]*100:.2f}%)\n")
            f.write(f"Fastest Training: {fastest_model[0]} ({fastest_model[1]['training_time']:.2f}s)\n")
            
            cnn_models = {k: v for k, v in results.items() if 'CNN' in k}
            if cnn_models:
                best_cnn = max(cnn_models.items(), key=lambda x: x[1]['accuracy'])
                f.write(f"Best CNN Model: {best_cnn[0]} ({best_cnn[1]['accuracy']*100:.2f}%)\n")
            
            svm_models = {k: v for k, v in results.items() if 'SVM' in k}
            if svm_models:
                best_svm = max(svm_models.items(), key=lambda x: x[1]['accuracy'])
                f.write(f"Best SVM Kernel: {best_svm[0]} ({best_svm[1]['accuracy']*100:.2f}%)\n")
            
            f.write("\n" + "="*70 + "\n")
        
        print(f"\nGenerated performance summary: {output_path}")


def main():
    print("="*60)
    print("Quick Doodle Classifier - Mini ML Project")
    print("="*60)
    
    dataset = DoodleDataset()
    
    X, y = dataset.load_data()
    print(f"\nDataset loaded: {X.shape[0]} images, {len(CLASSES)} classes")
    
    (X_train, y_train), (X_val, y_val), (X_test, y_test) = dataset.split_data(X, y)
    print(f"Train: {len(X_train)}, Val: {len(X_val)}, Test: {len(X_test)}")
    
    dataset.visualize_samples(X_train, y_train)
    
    X_train_processed = dataset.preprocess_data(X_train, binarize=True)
    X_val_processed = dataset.preprocess_data(X_val, binarize=True)
    X_test_processed = dataset.preprocess_data(X_test, binarize=True)
    
    evaluator = Evaluator(CLASSES)
    results = {}
    test_results = {}
    
    lr_model = LogisticRegressionModel(len(CLASSES))
    lr_time = lr_model.train(X_train_processed, y_train)
    lr_acc, lr_pred = lr_model.evaluate(X_val_processed, y_val)
    lr_test_acc, lr_test_pred = lr_model.evaluate(X_test_processed, y_test)
    
    results['Logistic Regression'] = {
        'accuracy': lr_acc,
        'training_time': lr_time,
        'predictions': lr_pred
    }
    test_results['Logistic Regression'] = lr_test_acc
    
    print(f"\nLogistic Regression Validation Accuracy: {lr_acc*100:.2f}%")
    evaluator.plot_confusion_matrix(y_val, lr_pred, 
                                    'Logistic Regression - Confusion Matrix',
                                    'confusion_matrix_lr.png')
    evaluator.print_classification_report(y_val, lr_pred, 'Logistic Regression')
    
    for kernel in ['linear', 'rbf', 'poly']:
        svm = SVMModel(len(CLASSES), kernel=kernel)
        svm_time = svm.train(X_train_processed, y_train)
        svm_acc, svm_pred = svm.evaluate(X_val_processed, y_val)
        svm_test_acc, svm_test_pred = svm.evaluate(X_test_processed, y_test)
        
        model_name = f'SVM ({kernel.capitalize()})'
        results[model_name] = {
            'accuracy': svm_acc,
            'training_time': svm_time,
            'predictions': svm_pred
        }
        test_results[model_name] = svm_test_acc
        
        print(f"\n{model_name} Validation Accuracy: {svm_acc*100:.2f}%")
        evaluator.plot_confusion_matrix(y_val, svm_pred,
                                       f'{model_name} - Confusion Matrix',
                                       f'confusion_matrix_svm_{kernel}.png')
        evaluator.print_classification_report(y_val, svm_pred, model_name)
    
    for version in ['v1', 'v2', 'v3']:
        cnn = SimpleCNN(len(CLASSES), version=version)
        cnn_time = cnn.train(X_train_processed, y_train, 
                            X_val_processed, y_val, 
                            epochs=15, batch_size=128)
        cnn_acc, cnn_pred = cnn.evaluate(X_val_processed, y_val)
        cnn_test_acc, cnn_test_pred = cnn.evaluate(X_test_processed, y_test)
        
        model_name = f'CNN {version}'
        results[model_name] = {
            'accuracy': cnn_acc,
            'training_time': cnn_time,
            'predictions': cnn_pred
        }
        test_results[model_name] = cnn_test_acc
        
        print(f"\nCNN {version} Validation Accuracy: {cnn_acc*100:.2f}%")
        cnn.plot_history(f'training_history_{version}.png')
        evaluator.plot_confusion_matrix(y_val, cnn_pred,
                                       f'CNN {version} - Confusion Matrix',
                                       f'confusion_matrix_{version}.png')
        evaluator.print_classification_report(y_val, cnn_pred, f'CNN {version}')
    
    best_model_name = max(results.items(), key=lambda x: x[1]['accuracy'])[0]
    best_predictions = results[best_model_name]['predictions']
    evaluator.plot_misclassified_samples(X_val_processed, y_val, best_predictions,
                                        'misclassified_samples.png', n_samples=10)
    
    evaluator.plot_comparison(results, 'model_comparison.png')
    
    evaluator.generate_summary_report(results, test_results, 'performance_summary.txt')
    
    print("\n" + "="*60)
    print("FINAL RESULTS SUMMARY")
    print("="*60)
    print(f"{'Model':<25} {'Val Accuracy':<15} {'Test Accuracy':<15} {'Time'}")
    print("-"*60)
    for model_name in results.keys():
        val_acc = results[model_name]['accuracy'] * 100
        test_acc = test_results[model_name] * 100
        time_taken = results[model_name]['training_time']
        print(f"{model_name:<25} {val_acc:>6.2f}%        {test_acc:>6.2f}%        {time_taken:>8.2f}s")
    
    best_model = max(results.items(), key=lambda x: x[1]['accuracy'])
    best_test_model = max(test_results.items(), key=lambda x: x[1])
    print(f"\nBest Validation Model: {best_model[0]} with {best_model[1]['accuracy']*100:.2f}% accuracy")
    print(f"Best Test Model: {best_test_model[0]} with {best_test_model[1]*100:.2f}% accuracy")
    
    print("\n" + "="*60)

if __name__ == "__main__":
    main()