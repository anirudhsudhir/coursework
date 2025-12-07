## 📁 File Structure

```
doodle-classifier/
│
├── main.py                    # Full implementation with all models
├── quickstart.py              # Quick test version (3 classes, 5 epochs)
├── requirements.txt           # Python dependencies
├── setup.sh                   # Linux/Mac setup script
├── setup.bat                  # Windows setup script
├── README.md                  # Project documentation
├── PROJECT_GUIDE.md          # This file
├── VIVA_GUIDE.md             # Demo & viva preparation
├── TECHNICAL_REPORT.md       # Complete technical report
├# Quick Doodle Classifier - Complete Project Guide

## 📁 File Structure

```

doodle-classifier/
│
├── main.py # Full implementation with all models
├── quickstart.py # Quick test version (3 classes, 5 epochs)
├── requirements.txt # Python dependencies
├── setup.sh # Linux/Mac setup script
├── setup.bat # Windows setup script
├── README.md # Project documentation
├── PROJECT_GUIDE.md # This file
│
├── data/ # Auto-created for datasets
│ ├── cat.npy # ~30MB each
│ ├── tree.npy
│ ├── car.npy
│ ├── apple.npy
│ └── fish.npy
│
└── venv/ # Python virtual environment

````

## 🚀 Quick Start Options

### Option 1: Quick Test (Recommended First)
```bash
chmod +x setup.sh
./setup.sh
source venv/bin/activate
python quickstart.py
````

**What it does:**

- Uses only 3 classes (cat, car, apple)
- 2,000 samples per class
- Trains only Logistic Regression + 1 simple CNN
- Takes ~5 minutes total
- Perfect for testing setup

### Option 2: Full Project

```bash
python main.py
```

**What it does:**

- Uses 5 classes (cat, tree, car, apple, fish)
- 5,000 samples per class
- Trains 4 models (LR + 3 CNN variants)
- Takes ~40-60 minutes
- Generates comprehensive results

## 📊 Code Architecture

### main.py Structure

```python
DoodleDataset class
├── load_data()           # Downloads and loads .npy files
├── split_data()          # 70/15/15 train/val/test split
├── preprocess_data()     # Binarization and normalization
└── visualize_samples()   # Creates sample images

LogisticRegressionModel class
├── train()               # Trains sklearn LogisticRegression
└── evaluate()            # Returns accuracy and predictions

SimpleCNN class
├── _build_model()        # Builds v1/v2/v3 architectures
├── train()               # Trains with Keras
├── evaluate()            # Returns accuracy and predictions
└── plot_history()        # Plots training curves

Evaluator class
├── plot_confusion_matrix()     # Creates confusion matrix heatmap
├── print_classification_report() # Prints precision/recall/F1
└── plot_comparison()           # Compares all models

main()
└── Orchestrates entire pipeline
```

### Model Architectures

**Logistic Regression:**

- Input: 784 features (28×28 flattened)
- Output: 5 classes (softmax)
- Parameters: ~4,000

**CNN v1 (Full):**

```
Input (28×28×1)
    ↓
Conv2D(32 filters, 3×3) + ReLU
    ↓
MaxPooling2D(2×2)
    ↓
Conv2D(64 filters, 3×3) + ReLU
    ↓
MaxPooling2D(2×2)
    ↓
Flatten
    ↓
Dense(128) + ReLU
    ↓
Dropout(0.25)
    ↓
Dense(5) + Softmax
```

Parameters: ~180,000

**CNN v2 (Simplified):**

```
Input (28×28×1)
    ↓
Conv2D(32 filters, 3×3) + ReLU
    ↓
MaxPooling2D(2×2)
    ↓
Flatten
    ↓
Dense(128) + ReLU
    ↓
Dropout(0.25)
    ↓
Dense(5) + Softmax
```

Parameters: ~130,000

**CNN v3 (Minimal):**

```
Input (28×28×1)
    ↓
Conv2D(32 filters, 3×3) + ReLU
    ↓
MaxPooling2D(2×2)
    ↓
Flatten
    ↓
Dense(64) + ReLU
    ↓
Dropout(0.25)
    ↓
Dense(5) + Softmax
```

Parameters: ~65,000

## 🎯 Expected Results

### Performance Benchmarks

| Model               | Accuracy | Training Time | Parameters |
| ------------------- | -------- | ------------- | ---------- |
| Logistic Regression | 60-70%   | 30-60s        | ~4K        |
| CNN v1 (Full)       | 85-90%   | 15-20 min     | ~180K      |
| CNN v2 (Simplified) | 82-88%   | 10-15 min     | ~130K      |
| CNN v3 (Minimal)    | 80-85%   | 8-12 min      | ~65K       |

### Key Insights

1. **CNN vs Classical ML**: CNNs significantly outperform Logistic Regression for image data
2. **Simplification**: Removing layers reduces training time with minimal accuracy loss
3. **Sweet Spot**: CNN v2 offers the best accuracy/speed tradeoff
4. **Binarization**: Reduces data size by 50% with only ~2-3% accuracy loss

## 📈 Generated Outputs

### Visualization Files

1. **sample_doodles.png** - Grid of example images from each class
2. **confusion_matrix_lr.png** - Shows which classes LR confuses
3. **confusion_matrix_v1/v2/v3.png** - CNN confusion matrices
4. **training_history_v1/v2/v3.png** - Loss and accuracy curves over epochs
5. **model_comparison.png** - Side-by-side accuracy and time comparison

## 🔧 Customization Guide

### Change Number of Classes

**In main.py:**

```python
CLASSES = ['cat', 'tree', 'car']
```

Available classes from QuickDraw:

- Animals: cat, dog, bird, fish, elephant, horse, sheep, etc.
- Objects: car, tree, house, chair, table, bed, etc.
- Food: apple, banana, pizza, cake, bread, etc.

### Reduce Dataset Size (Faster Testing)

```python
SAMPLES_PER_CLASS = 2000
```

### Modify Training Parameters

```python
cnn.train(X_train, y_train, X_val, y_val,
          epochs=15,
          batch_size=64)
```

### Add Custom CNN Architecture

```python
elif self.version == 'v4':
    model.add(layers.Conv2D(64, (3, 3), activation='relu',
                           input_shape=(IMG_SIZE, IMG_SIZE, 1)))
    model.add(layers.BatchNormalization())
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Conv2D(128, (3, 3), activation='relu'))
    model.add(layers.BatchNormalization())
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Flatten())
    model.add(layers.Dense(256, activation='relu'))
    model.add(layers.Dropout(0.3))
    model.add(layers.Dense(self.num_classes, activation='softmax'))
```

## 🐛 Troubleshooting

### Issue: Out of Memory

**Solution:**

```python
SAMPLES_PER_CLASS = 2000
batch_size = 32
```

### Issue: TensorFlow Warnings

Normal warnings you can ignore:

```
2024-01-15 10:30:45.123456: I tensorflow/core/...
```

Critical errors start with `ERROR:`

### Issue: Slow Download

**Solution:** Download manually

```bash
cd data
wget https://storage.googleapis.com/quickdraw_dataset/full/numpy_bitmap/cat.npy
wget https://storage.googleapis.com/quickdraw_dataset/full/numpy_bitmap/car.npy
...
```

### Issue: Module Not Found

**Solution:**

```bash
pip install --upgrade -r requirements.txt
```

## 📝 Assignment/Report Template

### Suggested Sections

1. **Introduction**

   - Motivation for doodle classification
   - Real-world applications
   - Project goals

2. **Dataset**

   - Google QuickDraw description
   - Selected classes and why
   - Preprocessing steps
   - Train/val/test split rationale

3. **Methods**

   - Logistic Regression explanation
   - CNN architecture descriptions
   - Training hyperparameters
   - Evaluation metrics

4. **Results**

   - Performance table (all models)
   - Confusion matrices analysis
   - Training curves discussion
   - Error analysis

5. **Discussion**

   - Accuracy vs efficiency tradeoff
   - Why CNNs outperform LR
   - Effect of simplification
   - Comparison with paper findings

6. **Conclusion**
   - Best model for this task
   - Key learnings
   - Future improvements

### Key Figures to Include

- Sample doodles from each class
- Model comparison bar charts
- Best model's confusion matrix
- Training history plot
- Accuracy vs training time scatter plot

## 🎓 Learning Objectives

By completing this project, you will understand:

1. ✅ End-to-end ML pipeline (data → training → evaluation)
2. ✅ Classical ML vs Deep Learning comparison
3. ✅ CNN architecture design principles
4. ✅ Model simplification and efficiency
5. ✅ Performance evaluation and visualization
6. ✅ Tradeoffs between accuracy and computational cost

## 📚 Further Reading

- Original Paper: "Drawing: A New Way To Search" (Stanford CS229)
- QuickDraw Dataset: https://github.com/googlecreativelab/quickdraw-dataset
- CNN Tutorial: https://cs231n.github.io/convolutional-networks/
- Keras Documentation: https://keras.io/guides/
