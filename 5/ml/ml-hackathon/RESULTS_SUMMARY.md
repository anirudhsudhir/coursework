# Hangman Agent Versions - Complete Results Summary

## Overview

This document summarizes all 12 versions of the Hangman AI agent, comparing approaches and results.

---

## Results Comparison Table

| Version | Approach                   | Success Rate | Avg Wrong | Score       | Key Features                          |
| ------- | -------------------------- | ------------ | --------- | ----------- | ------------------------------------- |
| V1      | Q-Learning + Basic HMM     | 30.2%        | 5.6       | -56,492     | Initial RL attempt, sparse Q-table    |
| V2      | Improved HMM               | 30.8%        | 5.5       | -55,266     | Better probability normalization      |
| V3      | Vowel-First Heuristic      | 28.9%        | 6.1       | -57,820     | **Worst** - Forced exploration failed |
| V4      | Pattern Matching           | 0%           | 6.0       | Failed      | Zero overlap killed word matching     |
| V5      | Statistical Patterns       | 32.1%        | 5.3       | -52,389     | Position frequency analysis           |
| V6      | Pure HMM Forward-Backward  | 30.5%        | 5.5       | -55,510     | Proper HMM without N-grams            |
| V7      | HMM with Gamma             | 30.5%        | 5.5       | -55,505     | Refined probability calculation       |
| **V8**  | **HMM + N-grams**          | **32.85%**   | **5.195** | **-51,288** | **BEST** - Combined approach          |
| V9      | Optimized Weights          | 32.6%        | 5.2       | -51,883     | Weight tuning                         |
| V10     | Final Tuning               | 32.7%        | 5.15      | -51,552     | Smoothing adjustments                 |
| V11     | Q-Learning (Feature-Based) | 0.25%        | 5.99      | -59,935     | Proper RL, but poor generalization    |
| V12     | SARSA + HMM Priors         | 13.3%        | 5.71      | -56,834     | On-policy RL with HMM guidance        |

---

## Detailed Analysis

### Best Performer: V8 (HMM + N-grams)

**Score: -51,288 | Success: 32.85%**

**Why it won:**

- Combines position-based HMM with contextual N-grams
- Adaptive weighting based on game state
- No repeated guesses (perfect efficiency)
- Best generalization to unseen words

**Architecture:**

```
Pattern → Forward-Backward (HMM) → Position probabilities
       ↓
       → N-grams (trigram/bigram/unigram) → Context probabilities
       ↓
       → Weighted combination → Final letter probabilities
       ↓
       → Greedy selection → Action
```

### Worst Performers

**V3 (Vowel-First): -57,820**

- Ignored learned patterns for fixed heuristic
- Words like "RHYTHM" have few vowels
- Demonstrates why data beats intuition

**V4 (Pattern Matching): Failed**

- Tried to match test words against corpus
- Zero overlap = zero matches
- Shows importance of generalization

**V11 (Q-Learning): -59,935**

- Proper RL implementation but poor results
- Feature-based state representation wasn't informative enough
- High exploration led to poor performance
- Needed more episodes or better features

### RL Implementations (V11, V12)

**V11: Q-Learning with Function Approximation**

- Success: 0.25% (5/2000 wins)
- Score: -59,935
- Problems:
  - Features didn't capture enough information
  - Random exploration hurt performance
  - Needed millions of episodes, not thousands
  - State abstraction too aggressive

**V12: SARSA with HMM Priors**

- Success: 13.3% (266/2000 wins)
- Score: -56,834
- Improvements over V11:
  - HMM priors for informed exploration
  - On-policy learning (more stable)
  - Better state hashing
  - 60% improvement in success rate vs V11

**Why RL Failed vs HMM:**

1. **Sample Efficiency**: RL needs millions of episodes; we had 50K training words
2. **Generalization**: Zero test-train overlap requires strong pattern learning
3. **State Space**: Even with abstraction, state space too large
4. **Reward Signal**: Sparse rewards (only at game end) make learning difficult
5. **Feature Engineering**: Hand-crafted features less powerful than HMM probabilities

---

## Key Lessons Learned

### 1. **Generalization > Memorization**

- V4's pattern matching failed completely
- V8's probabilistic patterns transferred to new words
- Zero overlap forces true pattern learning

### 2. **Context Matters**

- Pure position (V6): 30.5%
- Position + Context (V8): 32.85%
- N-grams capture letter co-occurrence

### 3. **Greedy Can Beat Exploration**

- V8 greedy: 32.85%
- V11 ε-greedy: 0.25%
- V12 informed exploration: 13.3%
- When you have good priors, exploit them

### 4. **Feature Engineering is Hard**

- V11's features didn't capture pattern nuances
- HMM's position-based emissions naturally informative
- Sometimes model-based > model-free

### 5. **RL Needs Data**

- 5,000-8,000 episodes insufficient
- Should need 100K+ for tabular methods
- Deep RL would need millions

---

## What Would Achieve Positive Score?

**Target: Score > 0**

Formula: `(Success × 2000) - (Wrong × 5) - (Repeated × 2) > 0`

If avg_wrong = 5.2:

- Need Success > (5.2 × 2000) / 2000 = **52% success rate**

**Current best: 32.85%**  
**Gap: +19.15% needed**

### Potential Improvements

**Short-term (+5-10%):**

1. Kneser-Ney smoothing instead of Laplace
2. Character-level RNN for pattern learning
3. Ensemble of multiple HMM variants
4. Better hyperparameter tuning

**Medium-term (+10-15%):**

1. Deep Q-Network with better features
2. Policy gradient methods
3. Transformer-based language model
4. Meta-learning for quick adaptation

**Long-term (+15-20%):**

1. Pre-trained LLM (GPT/BERT) fine-tuned on Hangman
2. Multi-task learning with related word games
3. Active learning during test time
4. Hybrid: Neural + HMM combination

---

## Proper RL vs Heuristic Policy

### V8 (Heuristic): Framework Only

```python
✓ Environment (HangmanGame)
✓ State representation
✓ Action space
✓ Reward function
✗ Learning algorithm (no parameter updates from rewards)
✗ Exploration strategy
```

**Verdict:** RL _framework_ with fixed policy, not true RL

### V11/V12: True RL

```python
✓ Environment
✓ State representation
✓ Action space
✓ Reward function
✓ Learning algorithm (Q-Learning / SARSA)
✓ Exploration strategy (ε-greedy)
✓ Parameter updates from experience
```

**Verdict:** Proper RL, but poor performance on this task

---

## Recommendations

### For Best Score (Competition)

**Use V8**: Proven best results, reliable, interpretable

### For Learning RL (Educational)

**Use V12**: Demonstrates proper RL concepts, shows learning curve

### For Research (Innovation)

**Try Deep RL**: DQN with better state encoding, more episodes

### For Production (Real-world)

**Ensemble**: Combine V8 (HMM) + V12 (SARSA) + Neural model

---

## Conclusion

The best performing agent (V8) uses a **hybrid probabilistic approach** combining:

1. HMM for position-based patterns
2. N-grams for contextual patterns
3. Adaptive weighting for game-state optimization
4. Greedy exploitation of learned probabilities

Pure RL methods (V11, V12) struggled due to:

1. Limited training data (50K words)
2. Zero train-test overlap
3. Large state space
4. Sparse reward signal

**Key Insight:** For structured problems with good priors (like Hangman), model-based methods (HMM) outperform model-free RL when data is limited. RL shines when you need to discover optimal strategies through pure interaction, but Hangman's structure is well-captured by statistical language models.

---

**Final Rankings:**

1. 🥇 V8: -51,288 (HMM + N-grams)
2. 🥈 V10: -51,552 (Tuned HMM)
3. 🥉 V9: -51,883 (Optimized weights)
4. V5: -52,389 (Statistical patterns)
5. V2: -55,266 (Improved HMM)
6. V7: -55,505 (HMM gamma)
7. V6: -55,510 (Pure HMM)
8. V1: -56,492 (Q-Learning + HMM)
9. V12: -56,834 (SARSA) ✨ **Best RL**
10. V3: -57,820 (Vowel-first)
11. V11: -59,935 (Q-Learning)
12. V4: Failed (Pattern matching)
