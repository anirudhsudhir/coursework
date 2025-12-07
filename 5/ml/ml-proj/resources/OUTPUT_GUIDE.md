# Output Directory Guide

All project outputs are organized in the `output/` directory for easy access and presentation.

## 📁 Directory Structure

```
output/
├── sample_doodles.png              # Dataset visualization
├── confusion_matrix_lr.png         # Logistic Regression
├── confusion_matrix_svm_linear.png # SVM Linear kernel
├── confusion_matrix_svm_rbf.png    # SVM RBF kernel
├── confusion_matrix_svm_poly.png   # SVM Polynomial kernel
├── confusion_matrix_v1.png         # CNN v1 (Full)
├── confusion_matrix_v2.png         # CNN v2 (Simplified)
├── confusion_matrix_v3.png         # CNN v3 (Minimal)
├── training_history_v1.png         # CNN v1 training curves
├── training_history_v2.png         # CNN v2 training curves
├── training_history_v3.png         # CNN v3 training curves
├── misclassified_samples.png       # Error analysis
├── model_comparison.png            # All models comparison
├── performance_summary.txt         # Comprehensive text report
└── quick_results.png               # (from quickstart.py only)
```

---

## 📊 File Descriptions

### 1. sample_doodles.png

**Purpose:** Dataset visualization  
**Content:** 10 example images (2 from each class)  
**Use in presentation:**

- Show dataset quality
- Explain drawing variability
- Demonstrate preprocessing

**What to discuss:**

- "Notice how drawings vary from detailed to rough sketches"
- "All images are 28×28 pixels, black and white"
- "This variability makes classification challenging"

---

### 2. confusion_matrix_lr.png

**Purpose:** Logistic Regression performance  
**Content:** 5×5 normalized confusion matrix  
**Use in presentation:**

- Baseline model performance
- Which classes are confused
- Limitations of linear models

**What to look for:**

- Diagonal brightness = accuracy
- Off-diagonal = confusions
- Typically: 60-70% accuracy

**Key insights:**

- "Linear model struggles with complex patterns"
- "Cat/Fish often confused due to curved shapes"
- "This establishes our baseline at ~65%"

---

### 3-5. confusion*matrix_svm*\*.png

**Purpose:** Compare SVM kernel performance  
**Content:** 3 confusion matrices (Linear, RBF, Poly)

**Files:**

- `confusion_matrix_svm_linear.png` - Linear kernel
- `confusion_matrix_svm_rbf.png` - RBF kernel (best)
- `confusion_matrix_svm_poly.png` - Polynomial kernel

**Use in presentation:**

- Show kernel comparison
- Demonstrate RBF superiority
- Bridge classical ML and deep learning

**Expected performance:**

- Linear: 55-65% (worst)
- RBF: 70-80% (best classical)
- Poly: 65-75% (medium)

**Key insights:**

- "RBF kernel maps to infinite dimensions"
- "Non-linear kernels essential for image data"
- "Still trails CNNs by ~10-15%"

---

### 6-8. confusion_matrix_v\*.png

**Purpose:** CNN performance analysis  
**Content:** 3 confusion matrices for CNN variants

**Files:**

- `confusion_matrix_v1.png` - Full model
- `confusion_matrix_v2.png` - Simplified
- `confusion_matrix_v3.png` - Minimal

**Use in presentation:**

- Show CNN superiority
- Demonstrate simplification impact
- Analyze per-class performance

**What to highlight:**

- Much brighter diagonal than classical ML
- Fewer off-diagonal confusions
- Consistent high performance across classes

**Key insights:**

- "CNNs learn hierarchical features automatically"
- "Even simplified CNN v3 beats best SVM"
- "Best model achieves xyz% accuracy"

---

### 9-11. training_history_v\*.png

**Purpose:** Training dynamics visualization  
**Content:** Accuracy and loss curves over epochs

**Each plot shows:**

- Left: Training vs Validation Accuracy
- Right: Training vs Validation Loss

**Use in presentation:**

- Show convergence behavior
- Demonstrate no overfitting
- Explain early stopping

**What to look for:**

- Train/val curves stay close = good generalization
- Curves plateau = convergence
- Early stopping triggers when no improvement

**Key insights:**

- "Model converges around epoch 8-10"
- "Small train-val gap shows no overfitting"
- "Early stopping saved ~5 epochs of training"

---

### 12. misclassified_samples.png

**Purpose:** Error analysis  
**Content:** 10 examples where best model failed

**Shows:**

- True label vs Predicted label
- Visual examples of mistakes

**Use in presentation:**

- Analyze failure modes
- Discuss model limitations
- Show human-level difficulty

**Common error patterns:**

- Ambiguous/rough sketches
- Unusual perspectives
- Shape similarities (cat ↔ fish)

**Key insights:**

- "Some errors are understandable even for humans"
- "Drawing quality significantly affects accuracy"
- "Model learned reasonable features"

---

### 13. model_comparison.png

**Purpose:** Overall performance comparison  
**Content:** Two bar charts

**Left chart:** Accuracy comparison (%)
**Right chart:** Training time comparison (seconds)

**Use in presentation:**

- Show complete picture
- Demonstrate tradeoffs
- Guide model selection

**Key insights:**

- "CNNs dominate accuracy (80-88%)"
- "Classical ML is faster but less accurate"
- "CNN v3 offers best efficiency/accuracy ratio"

---

### 14. performance_summary.txt

**Purpose:** Comprehensive text report  
**Content:** All metrics in organized format

**Sections:**

1. Dataset information
2. Validation set results
3. Test set results
4. Key findings
5. Observations

**Use in presentation:**

- Quick reference for numbers
- Include in written report
- Share with stakeholders

**Example content:**

```
Model                     Accuracy        Time (s)
Logistic Regression       65.2%           45s
SVM (RBF)                 74.6%           240s
CNN v1                    88.3%           900s
CNN v2                    85.1%           600s
CNN v3                    82.7%           480s
```

---

## 🎯 Presentation Strategy

### For 5-Minute Demo

1. Show `sample_doodles.png` (30s)
2. Show `model_comparison.png` (2min)
3. Show best `confusion_matrix_v1.png` (1.5min)
4. Show `misclassified_samples.png` (1min)

### For 10-Minute Demo

Add: 5. Show `training_history_v2.png` (2min) 6. Show SVM comparison (2min) 7. Discuss `performance_summary.txt` (1min)

### For 15-Minute Demo

Add: 8. Detailed architecture explanation with diagrams 9. Show all confusion matrices 10. Discuss future improvements

---

## 📋 Using in Written Report

**Figures to include:**

1. **Introduction section:**

   - `sample_doodles.png` as Figure 1

2. **Methodology section:**

   - CNN architecture diagram (create manually)
   - SVM kernel comparison (create from theory)

3. **Results section:**

   - `model_comparison.png` as main results figure
   - `confusion_matrix_v1.png` for best model
   - `training_history_v1.png` for convergence

4. **Discussion section:**

   - `misclassified_samples.png` for error analysis
   - All confusion matrices in appendix

5. **Appendix:**
   - Include `performance_summary.txt` content
   - All other confusion matrices
   - All training history plots

---

## 🎨 Creating Presentation Slides

### Suggested Slide Breakdown

**Slide 1: Title**

- Project name
- Your name/team
- Date

**Slide 2: Dataset**

- Image: `sample_doodles.png`
- Bullet points about dataset

**Slide 3: Models Overview**

- Table of 7 models
- Brief description each

**Slide 4: Results Overview**

- Image: `model_comparison.png`
- Highlight best performers

**Slide 5: CNN Performance**

- Image: `confusion_matrix_v1.png`
- Per-class accuracy discussion

**Slide 6: Training Dynamics**

- Image: `training_history_v2.png`
- Convergence explanation

**Slide 7: Error Analysis**

- Image: `misclassified_samples.png`
- Discussion of failure modes

**Slide 8: Conclusion**

- Key findings
- Best model recommendation
- Future work

---

## 🔧 Customizing Outputs

### Change Output Directory

Edit in `main.py`:

```python
OUTPUT_DIR = Path('my_results')
```

### Increase Figure Quality

Edit DPI in save calls:

```python
plt.savefig(output_path, dpi=300, bbox_inches='tight')
```

### Add More Visualizations

Add in `Evaluator` class:

```python
def plot_custom_metric(self, ...):
    # Your custom plot
    output_path = OUTPUT_DIR / 'custom_plot.png'
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
```

---

## 📊 Understanding Each Visualization

### Confusion Matrix

- **Bright diagonal** = Good accuracy
- **Off-diagonal brightness** = Common confusions
- **Row normalization** = Each row sums to 1.0
- **Color scale** = 0.0 (white) to 1.0 (dark blue)

### Training History

- **Y-axis** = Accuracy/Loss
- **X-axis** = Epoch number
- **Blue line** = Training set
- **Orange line** = Validation set
- **Gap between lines** = Overfitting indicator

### Bar Charts

- **Height** = Metric value
- **Labels** = Exact numbers
- **Colors** = Distinguish models
- **Grid** = Easy value reading

---

## 🎓 Common Questions About Outputs

**Q: Why are some confusion matrices darker than others?**
A: Darker diagonals = higher accuracy. CNNs have darkest diagonals (85-90%), classical ML lighter (60-75%).

**Q: What if train/val curves diverge in training history?**
A: Divergence = overfitting. Our curves stay close, showing good generalization.

**Q: Why is misclassified_samples.png important?**
A: Shows model isn't just a black box. Helps understand failure modes and guide improvements.

**Q: Should I include all files in my presentation?**
A: No, select 4-5 most important. Use others for detailed questions or written report.

**Q: How do I reference these in my report?**
A: "As shown in Figure 3 (model_comparison.png), CNNs significantly outperform classical approaches..."

---
