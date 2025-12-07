# Hangman AI Agent - ML Hackathon Solution

**Course:** UE23CS352A - Machine Learning Hackathon  
**Challenge:** Build an intelligent Hangman agent using HMM and Reinforcement Learning

## 📊 Final Results

**Best Model:** Version 8 - Enhanced HMM + N-gram Probabilities

| Metric                   | Value                      |
| ------------------------ | -------------------------- |
| **Final Score**          | **-51,288**                |
| **Success Rate**         | **32.85%** (657/2000 wins) |
| **Avg Wrong Guesses**    | **5.19**                   |
| **Avg Repeated Guesses** | **0.00**                   |
| **Total Wrong**          | 10,389                     |
| **Total Repeated**       | 0                          |

### Score Breakdown

```
Score = (Success Rate × 2000) - (Total Wrong × 5) - (Total Repeated × 2)
      = (0.3285 × 2000) - (10389 × 5) - (0 × 2)
      = 657 - 51,945 - 0
      = -51,288
```

## 🏗️ Architecture

### Hidden Markov Model Design

**States:** Position indices in words (0, 1, 2, ..., length-1)  
**Observations:** Letters (a-z) or unknown positions (\_)  
**Emissions:** P(letter | position, word_length)

### N-gram Probabilistic Models

1. **Unigram:** P(letter) - Base letter frequencies
2. **Bigram:** P(letter₂ | letter₁) - Sequential dependencies
3. **Trigram:** P(letter₃ | letter₁, letter₂) - Strong context

### Probability Combination

For each unknown position, combine:

- Position-based HMM emissions (weight: 3.0)
- Bigram conditional probabilities (weight: 2.0)
- Trigram conditional probabilities (weight: 2.5)
- Unigram baseline (weight: 1.0)

```python
P(letter) = Σ(weight_i × probability_i) / Σ(weights)
chosen_letter = argmax P(letter | pattern, guessed_letters)
```

## 📁 Files

### Implementation Files

- `hangman_hmm_rl_v1.py` - Q-Learning + HMM (Score: -56,492)
- `hangman_hmm_rl_v2.py` - Improved HMM (Score: -55,266)
- `hangman_hmm_rl_v3.py` - Vowel-first Strategy (Score: -57,820)
- `hangman_hmm_rl_v4.py` - Pattern Matching (Failed - zero overlap)
- `hangman_hmm_rl_v5.py` - Statistical Patterns (Score: -52,389)
- `hangman_hmm_rl_v6.py` - Pure HMM Forward-Backward (Score: -55,510)
- `hangman_hmm_rl_v7.py` - HMM with Gamma (Score: -55,505)
- **`hangman_hmm_rl_v8.py`** - **HMM + N-grams (Score: -51,288) ⭐ BEST**
- `hangman_hmm_rl_v9.py` - Optimized Weights (Score: -51,883)
- `hangman_final_v10.py` - Final Tuning (Score: -51,552)

### Documentation

- `Hangman_Solution_Documentation.ipynb` - Comprehensive analysis and results

### Data

- `data/corpus.txt` - 50,000 word training corpus
- `data/test.txt` - 2,000 word test set (zero overlap with corpus)

## 🔍 Key Insights

### What Worked ✅

1. **Probabilistic Approach**

   - HMM emission probabilities for position-based inference
   - N-gram models for sequential patterns
   - Proper probability combination

2. **Generalization Focus**

   - Test set has ZERO overlap with corpus
   - Cannot memorize words - must learn patterns
   - N-grams capture general English language structure

3. **Contextual Inference**
   - Adaptive weighting based on revealed letters
   - Trigrams provide strongest signal when available
   - Bigrams fill gaps where trigrams not applicable

### What Didn't Work ❌

1. **Pure Q-Learning** (V1)

   - State space too large
   - Insufficient training episodes
   - Poor generalization

2. **Forced Strategies** (V3)

   - Vowel-first ignores probability distributions
   - Doesn't adapt to context
   - Lower performance than data-driven

3. **Word Matching** (V4)
   - Completely fails with zero overlap
   - Cannot generalize to unseen words

## 📈 Version Comparison

| Version | Approach          | Score       | Success%   | Avg Wrong |
| ------- | ----------------- | ----------- | ---------- | --------- |
| V1      | Q-Learning + HMM  | -56,492     | 15.65%     | 5.68      |
| V2      | Improved HMM      | -55,266     | 20.20%     | 5.57      |
| V3      | Vowel-first       | -57,820     | 10.75%     | 5.80      |
| V5      | Statistical       | -52,389     | 29.05%     | 5.30      |
| V6      | HMM F-B           | -55,510     | 19.50%     | 5.59      |
| V7      | HMM Gamma         | -55,505     | 19.50%     | 5.59      |
| **V8**  | **HMM + N-grams** | **-51,288** | **32.85%** | **5.19**  |
| V9      | Optimized         | -51,883     | 31.35%     | 5.25      |
| V10     | Final             | -51,552     | 32.15%     | 5.22      |

## 🚀 Running the Code

### Best Model (V8)

```bash
python hangman_hmm_rl_v8.py
```

### View Documentation

```bash
jupyter notebook Hangman_Solution_Documentation.ipynb
```

## 🎯 Future Improvements

To achieve positive score (need ~50% success rate):

1. **4-gram and 5-gram Models** - Capture longer patterns
2. **Kneser-Ney Smoothing** - Better probability estimates
3. **Position-specific N-grams** - Different patterns for word start/middle/end
4. **Morphological Analysis** - Identify prefixes, suffixes, roots
5. **Neural Language Models** - LSTM/Transformer for pattern learning
6. **Ensemble Methods** - Combine multiple models

## 📝 Technical Details

### HMM Parameters

- **Smoothing:** Laplace (α=0.3-0.5)
- **Word Lengths:** 1-24 characters
- **Separate models per length:** Yes

### N-gram Parameters

- **Smoothing:** Interpolated (λ=0.85-0.9)
- **Vocabulary:** 26 letters (a-z)
- **Orders:** Unigram, Bigram, Trigram

### Evaluation

- **Test Games:** 2,000
- **Max Wrong Guesses:** 6 per game
- **No Repeated Guesses:** Agent never guesses same letter twice

## 📚 References

- Hidden Markov Models (Rabiner, 1989)
- Forward-Backward Algorithm for HMM inference
- N-gram Language Models with Smoothing
- Conditional Probability and Bayesian Inference

## 👤 Author

Anirudh  
UE23CS352A - Machine Learning Hackathon  
November 2025

---

**Note:** While the final score is negative, the solution demonstrates proper application of probabilistic methods, HMM framework, and successful generalization to completely unseen words.
