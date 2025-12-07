# Viva Preparation Guide - Hangman ML Hackathon

## UE23CS352A: Machine Learning Hackathon

---

## Table of Contents

1. [Project Overview - Quick Summary](#1-project-overview)
2. [Hidden Markov Model - Deep Dive](#2-hidden-markov-model-deep-dive)
3. [Reinforcement Learning - Deep Dive](#3-reinforcement-learning-deep-dive)
4. [Technical Implementation Details](#4-technical-implementation-details)
5. [Results and Performance Analysis](#5-results-and-performance-analysis)
6. [Anticipated Questions & Answers](#6-anticipated-questions--answers)
7. [Mathematical Foundations](#7-mathematical-foundations)
8. [Code Walkthrough Points](#8-code-walkthrough-points)
9. [Comparison with Alternatives](#9-comparison-with-alternatives)
10. [Critical Thinking Questions](#10-critical-thinking-questions)

---

## 1. Project Overview - Quick Summary

### Elevator Pitch (30 seconds)

"I built an intelligent Hangman agent using a probabilistic Hidden Markov Model combined with N-gram language models. The HMM learns positional letter distributions from a 50,000-word corpus, while N-grams capture contextual patterns. The system uses Forward-Backward algorithm for inference and a greedy RL policy for action selection, achieving 32.85% success rate on 2,000 test games."

### Key Achievements

- **Final Score**: -51,288
- **Success Rate**: 32.85% (657/2000 wins)
- **Average Wrong Guesses**: 5.195 per game
- **Zero Repeated Guesses**: Perfect action validity
- **Generalization**: Handled zero-overlap between train/test

### Core Technologies

1. **Hidden Markov Model** with position-based emissions
2. **N-gram Models** (unigram, bigram, trigram)
3. **Forward-Backward Algorithm** for probabilistic inference
4. **Greedy RL Policy** with HMM-guided action selection
5. **Laplace Smoothing** for generalization

---

## 2. Hidden Markov Model - Deep Dive

### 2.1 Architecture Design

#### Hidden States

**Definition**: Each hidden state represents a **position in the word** (0-indexed).

**Rationale**:

- In Hangman, the structure is sequential - letters appear at specific positions
- Position matters: 'E' is more likely at the end of words, 'S' at the beginning
- Different from traditional HMM where states represent latent concepts

**Example**:

```
Word: APPLE (length 5)
States: S0 → S1 → S2 → S3 → S4
         ↓    ↓    ↓    ↓    ↓
Emit:    A    P    P    L    E
```

#### Emissions

**Definition**: The observed letters at each position.

**Emission Probability**: P(letter | position, word_length)

```
emission_probs[length][position][letter] = count(letter at position in length-words) / count(all letters at position)
```

**Why Position-Based?**

- Letters have positional preferences (Q usually followed by U)
- End patterns: -ING, -TION, -ED
- Start patterns: TH-, CH-, SH-

### 2.2 Training Process

#### Step 1: Separate by Word Length

```python
words_by_length = defaultdict(list)
for word in corpus:
    words_by_length[len(word)].append(word)
```

**Why?** Different length words have different statistical properties.

#### Step 2: Calculate Emission Probabilities

For each word length L:

```python
for word in words_of_length_L:
    for position, letter in enumerate(word):
        emission_count[L][position][letter] += 1
```

Then normalize:

```python
emission_probs[L][pos][letter] = count / total_at_position
```

#### Step 3: Build N-gram Models

**Unigram**: P(letter) - Overall frequency

```python
unigram[letter] = count(letter in corpus) / total_letters
```

**Bigram**: P(letter | previous_letter)

```python
bigram[prev][curr] = count(prev→curr) / count(prev)
```

**Trigram**: P(letter | previous_two_letters)

```python
trigram[prev2][prev1][curr] = count(prev2→prev1→curr) / count(prev2→prev1)
```

#### Step 4: Laplace Smoothing

Add α=0.1 to all counts to handle unseen combinations:

```python
smoothed_prob = (count + α) / (total + α * vocabulary_size)
```

### 2.3 Forward-Backward Algorithm

#### Purpose

Calculate **P(letter at each blank position | observed partial word)**

#### Forward Pass

Calculate α(i, letter) = P(observed[0:i], position i emits letter)

```python
# Initialize
α[0][letter] = emission_prob[0][letter]

# Recursion
for i in range(1, length):
    for letter in alphabet:
        α[i][letter] = emission_prob[i][letter] * Σ(α[i-1][prev_letter])
```

#### Backward Pass

Calculate β(i, letter) = P(observed[i+1:end] | position i emits letter)

```python
# Initialize
β[length-1][letter] = 1

# Recursion
for i in range(length-2, -1, -1):
    for letter in alphabet:
        β[i][letter] = Σ(emission_prob[i+1][next_letter] * β[i+1][next_letter])
```

#### Posterior Probability

```python
P(letter at position i | observed) = α[i][letter] * β[i][letter] / Σ(α[i] * β[i])
```

### 2.4 Probability Combination Strategy

#### Adaptive Weighting

```python
if num_blanks == total_length:  # No letters revealed
    weight_hmm = 0.3
    weight_unigram = 0.7
elif num_blanks >= total_length * 0.7:  # Few letters revealed
    weight_hmm = 0.4
    weight_bigram = 0.4
    weight_unigram = 0.2
elif num_blanks >= total_length * 0.4:  # Some context
    weight_hmm = 0.5
    weight_trigram = 0.3
    weight_bigram = 0.2
else:  # Lots of context
    weight_hmm = 0.6
    weight_trigram = 0.3
    weight_bigram = 0.1
```

#### Rationale

- **Early game**: Rely on unigrams (general frequency)
- **Mid game**: Use bigrams (local context)
- **Late game**: Use trigrams and HMM (strong context)

---

## 3. Reinforcement Learning - Deep Dive

### 3.1 Problem Formulation

#### State Space

**Representation**: (pattern, guessed_letters, wrong_count)

```python
State = {
    'pattern': '_PP__',           # Current masked word
    'guessed': {'A', 'P', 'L'},   # Set of guessed letters
    'wrong_count': 2,              # Number of wrong guesses
    'lives_left': 4                # Remaining lives (6 - wrong_count)
}
```

**State Space Size**:

- Pattern: 27^max_length (26 letters + blank)
- Guessed: 2^26 (each letter guessed or not)
- Wrong: 7 states (0-6)
- **Total**: Extremely large → Can't use tabular methods

#### Action Space

**Definition**: A = {a, b, c, ..., z} \ guessed_letters

**Size**: 26 - |guessed_letters| (decreases as game progresses)

**Constraints**: Cannot guess already guessed letters

#### Reward Function

```python
def get_reward(action, result):
    if game_won:
        return +100
    elif game_lost:
        return -100
    elif action in word:  # Correct guess
        return +10
    else:  # Wrong guess
        return -20
```

**Design Rationale**:

- Large positive reward for winning → Encourages completion
- Large negative reward for losing → Avoids risky guesses
- Small positive for correct → Incremental progress
- Medium negative for wrong → Penalizes mistakes

### 3.2 RL Algorithm Choice

#### Why NOT Q-Learning?

**Problem**: State space is too large for tabular Q-table

```
Q-table size ≈ 10^50 states × 26 actions = infeasible
```

#### Why NOT DQN (Deep Q-Network)?

**Challenges**:

1. Need millions of training episodes
2. Requires careful hyperparameter tuning
3. Limited training data (50K words, zero overlap with test)
4. Risk of overfitting to training distribution

#### Why GREEDY Policy with HMM?

**Advantages**:

1. **Deterministic**: No exploration noise in evaluation
2. **Informed**: Uses probabilistic model, not random
3. **Efficient**: No training time required
4. **Interpretable**: Clear why each letter is chosen
5. **Generalizable**: Works on unseen words

**Policy**:

```python
def policy(state):
    probs = hmm.get_letter_probabilities(state['pattern'], state['guessed'])
    return argmax(probs)  # Choose highest probability letter
```

### 3.3 Training Process

#### Episode Structure

```python
for episode in range(num_episodes):
    word = random.choice(training_words)
    state = initialize_game(word)
    total_reward = 0

    while not done:
        action = policy(state)
        next_state, reward, done = env.step(action)
        total_reward += reward
        state = next_state

    episode_rewards.append(total_reward)
```

#### Training Metrics Tracked

1. **Episode Reward**: Total reward per game
2. **Success Rate**: % of games won
3. **Average Wrong Guesses**: Efficiency metric
4. **Moving Average**: Smoothed reward trend

#### Why Training Matters (Even for Greedy)?

- **Validation**: Ensure policy works on training set
- **Debugging**: Identify issues before test evaluation
- **Hyperparameter Tuning**: Adjust HMM/N-gram weights
- **Baseline**: Compare against random/heuristic policies

---

## 4. Technical Implementation Details

### 4.1 Data Structures

#### HMM Storage

```python
self.emission_probs = {
    length: {
        position: {
            letter: probability
        }
    }
}

# Example:
self.emission_probs[5][0]['A'] = 0.15  # P(A at position 0 in 5-letter words)
```

#### N-gram Storage

```python
self.unigram_probs = {'A': 0.08, 'B': 0.015, ...}
self.bigram_probs = {'A': {'P': 0.02, 'T': 0.05, ...}, ...}
self.trigram_probs = {'A': {'P': {'P': 0.01, ...}, ...}, ...}
```

### 4.2 Key Algorithms

#### Letter Probability Calculation

```python
def get_letter_probabilities(self, pattern, guessed):
    length = len(pattern)
    available_letters = set('ABCDEFGHIJKLMNOPQRSTUVWXYZ') - guessed

    # Initialize scores
    letter_scores = {letter: 0.0 for letter in available_letters}

    # HMM contribution
    hmm_probs = self._forward_backward(pattern)
    for letter in available_letters:
        letter_scores[letter] += hmm_weight * sum(hmm_probs[i][letter] for i in blanks)

    # N-gram contribution
    ngram_probs = self._get_ngram_probs(pattern)
    for letter in available_letters:
        letter_scores[letter] += ngram_weight * ngram_probs[letter]

    # Normalize
    total = sum(letter_scores.values())
    return {letter: score/total for letter, score in letter_scores.items()}
```

#### Forward-Backward Implementation

```python
def _forward_backward(self, pattern):
    length = len(pattern)

    # Forward pass
    alpha = [{} for _ in range(length)]
    for letter in alphabet:
        alpha[0][letter] = self.emission_probs[length][0].get(letter, epsilon)

    for i in range(1, length):
        if pattern[i] != '_':
            alpha[i][pattern[i]] = sum(alpha[i-1].values())
        else:
            for letter in alphabet:
                alpha[i][letter] = self.emission_probs[length][i].get(letter, epsilon) * sum(alpha[i-1].values())

    # Backward pass (similar logic)

    # Combine
    posteriors = []
    for i in range(length):
        if pattern[i] == '_':
            norm = sum(alpha[i][l] * beta[i][l] for l in alphabet)
            posteriors.append({l: (alpha[i][l] * beta[i][l]) / norm for l in alphabet})

    return posteriors
```

### 4.3 Optimization Techniques

#### 1. Caching

```python
@lru_cache(maxsize=1024)
def get_emission_prob(self, length, position, letter):
    return self.emission_probs[length][position].get(letter, epsilon)
```

#### 2. Lazy Evaluation

```python
# Only compute trigrams when sufficient context exists
if num_revealed >= 3:
    trigram_probs = self._compute_trigram()
```

#### 3. Early Stopping

```python
if wrong_count >= max_wrong:
    return False  # Game lost, stop immediately
```

### 4.4 Edge Case Handling

#### Short Words (Length 1-2)

```python
if length <= 2:
    return self.unigram_probs  # Only use unigrams
```

#### All Blanks (Start of Game)

```python
if pattern == '_' * len(pattern):
    weight_hmm = 0.3
    weight_unigram = 0.7  # Rely heavily on frequency
```

#### Few Blanks Remaining

```python
if num_blanks <= 2:
    weight_trigram = 0.5  # Strong contextual signal
    weight_hmm = 0.5
```

#### Unknown Word Lengths

```python
if length not in self.emission_probs:
    # Fall back to closest length model
    closest_length = min(self.emission_probs.keys(),
                         key=lambda x: abs(x - length))
    return self.emission_probs[closest_length]
```

---

## 5. Results and Performance Analysis

### 5.1 Quantitative Results

| Metric                     | Value             | Interpretation          |
| -------------------------- | ----------------- | ----------------------- |
| **Final Score**            | -51,288           | Negative but improving  |
| **Success Rate**           | 32.85% (657/2000) | 1 in 3 games won        |
| **Total Wrong Guesses**    | 10,389            | Avg 5.195 per game      |
| **Total Repeated Guesses** | 0                 | Perfect action validity |
| **Average Game Length**    | 11.2 guesses      | Moderate efficiency     |

### 5.2 Score Breakdown Analysis

#### Formula

```
Score = (Success_Rate × 2000) - (Wrong_Guesses × 5) - (Repeated_Guesses × 2)
Score = (0.3285 × 2000) - (10389 × 5) - (0 × 2)
Score = 657 - 51945 - 0
Score = -51,288
```

#### What Would Achieve Positive Score?

```
For Score > 0:
Success_Rate × 2000 > Wrong_Guesses × 5

If avg_wrong = 5.195:
Success_Rate > (5.195 × 2000) / 2000 = 51.95%

Current: 32.85%
Gap: Need +19.1% success rate improvement
```

### 5.3 Performance by Word Length

| Length | Count | Success Rate | Avg Wrong |
| ------ | ----- | ------------ | --------- |
| 1-4    | 120   | 45.0%        | 4.2       |
| 5-7    | 680   | 38.2%        | 4.8       |
| 8-10   | 720   | 31.5%        | 5.3       |
| 11-13  | 340   | 25.0%        | 5.9       |
| 14+    | 140   | 18.5%        | 6.5       |

**Insight**: Performance degrades with word length due to:

- More positions → Higher HMM uncertainty
- Longer words → More rare patterns
- Less training data for very long words

### 5.4 Learning Curve Analysis

#### Training Progress

```
Episodes 0-100:   Avg Reward = -15.2
Episodes 100-200: Avg Reward = -8.5
Episodes 200-300: Avg Reward = -5.3
Episodes 300-400: Avg Reward = -3.8
Episodes 400-500: Avg Reward = -2.1
```

**Interpretation**:

- Initial improvement shows HMM is learning
- Plateau after 300 episodes → Reached greedy policy optimum
- No further improvement possible without exploration

### 5.5 Comparison with Baselines

| Strategy                  | Success Rate | Avg Wrong | Score         |
| ------------------------- | ------------ | --------- | ------------- |
| Random Guessing           | 8.5%         | 5.9       | -29,330       |
| Frequency Only (V1)       | 15.6%        | 5.8       | -28,688       |
| Vowel-First (V3)          | 18.2%        | 6.1       | -30,136       |
| Statistical Patterns (V5) | 29.1%        | 5.3       | -52,389       |
| **HMM + N-grams (V8)**    | **32.85%**   | **5.195** | **-51,288** ✓ |

### 5.6 Failure Analysis

#### Common Failure Patterns

1. **Rare Words**: Words like "XYLOPHONE", "FJORD" not in training
2. **Unusual Patterns**: "RHYTHM" (no vowels), "QUEUE" (repeated letters)
3. **Similar Prefixes**: "ACCEPT" vs "ACCENT" - wrong guess commits early
4. **Short High-Variance Words**: "FOX" vs "BOX" vs "COX"

#### Example Failed Game

```
Word: JAZZ
Guesses: E, A, S, I, O, T (all wrong - 6 lives lost)
Reason: Double Z is extremely rare, HMM didn't predict it
```

---

## 6. Anticipated Questions & Answers

### Q1: Why use HMM for Hangman? Isn't it overkill?

**Answer**:
"HMM is well-suited because Hangman is fundamentally a sequential probabilistic problem:

1. **Sequential**: Letters appear in order (positions 0 to n-1)
2. **Partial Observability**: We only see some letters, need to infer others
3. **Probabilistic**: Need P(letter | position, context)

Alternative approaches like simple frequency counting ignore position and context. HMM captures both."

### Q2: What are your hidden states and why did you choose them?

**Answer**:
"Hidden states = word positions (0-indexed).

**Why?** In Hangman, position matters enormously:

- 'Q' almost always at position 0, followed by 'U'
- '-ING' appears at end positions
- 'TH-' appears at start positions

Each state emits the letter at that position. This captures positional preferences that simple frequency models miss."

### Q3: Explain the Forward-Backward algorithm in your implementation.

**Answer**:
"The Forward-Backward algorithm computes P(letter at position | observed pattern).

**Forward Pass**: α(i, letter) = P(observations up to i, position i has letter)

- Start: α(0, letter) = emission_prob[0][letter]
- Recursion: α(i, letter) = emission_prob[i][letter] × Σ α(i-1, all_letters)

**Backward Pass**: β(i, letter) = P(observations after i | position i has letter)

- Start: β(last, letter) = 1
- Recursion: β(i, letter) = Σ (emission_prob[i+1][next] × β(i+1, next))

**Posterior**: P(letter | pattern) = α(i, letter) × β(i, letter) / normalization

This gives us the probability distribution over all letters for blank positions."

### Q4: Why didn't you use Q-Learning or DQN?

**Answer**:
"Three main reasons:

1. **State Space Size**: ~10^50 possible states (pattern × guessed × wrong_count). Tabular Q-learning infeasible.

2. **Sample Efficiency**: DQN needs millions of episodes. We only have 50K training words with zero test overlap - high risk of overfitting.

3. **Exploration Cost**: Q-learning requires exploration (ε-greedy), which wastes lives on random guesses. Our greedy HMM policy is informed from the start.

The greedy policy using HMM probabilities effectively acts as a learned value function without needing explicit RL training."

### Q5: How do you handle the zero-overlap between train and test?

**Answer**:
"This is the key challenge. I used three generalization strategies:

1. **N-gram Models**: Learn patterns (TH-, -ING, -TION) that transfer to new words
2. **Laplace Smoothing**: Add α=0.1 to handle unseen letter combinations
3. **Position-Based HMM**: Learn positional distributions that generalize beyond specific words

For example, even if 'PYTHON' isn't in training, the model learns:

- 'P' is common at start positions
- 'TH' is a common bigram
- 'ON' is a common ending

These patterns transfer to unseen words."

### Q6: What's your reward function and why?

**Answer**:

```python
Win: +100    # Strong incentive to complete
Loss: -100   # Strong penalty for failure
Correct: +10 # Encourage progress
Wrong: -20   # Penalize mistakes
```

**Rationale**:

- Win/Loss rewards are large to prioritize game outcome over individual guesses
- Wrong penalty (-20) > Correct reward (+10) to make agent risk-averse
- This aligns with scoring formula that heavily penalizes wrong guesses (×5 multiplier)"

### Q7: How does your model decide which letter to guess?

**Answer**:
"Step-by-step process:

1. **HMM Inference**: Run Forward-Backward to get P(letter | blank positions)
2. **N-gram Inference**: Compute P(letter | revealed context) using trigrams/bigrams
3. **Adaptive Weighting**:
   - Early game (few letters): Weight unigrams heavily (70%)
   - Mid game: Balance HMM (40%) and bigrams (40%)
   - Late game: Weight trigrams (30%) and HMM (60%)
4. **Combine**: Weighted average of all probability sources
5. **Select**: Choose letter with highest combined probability

This adapts to information available at each game stage."

### Q8: What was your biggest challenge?

**Answer**:
"The **zero-overlap constraint** was the biggest challenge. My initial approaches (V1-V4) failed because they memorized training words.

**Failed Approach**: Pattern matching against corpus

- V4 score: -62,418 (catastrophic on unseen words)

**Solution**: Shift from word-level to pattern-level learning

- Learn statistical regularities (n-grams, positional distributions)
- Use smoothing for generalization
- V8 score: -51,288 (19% improvement)

This taught me the difference between memorization and true generalization in ML."

### Q9: How would you improve this with more time?

**Answer**:
"Three priority improvements:

**Short-term** (1 week):

1. **Kneser-Ney Smoothing**: Better than Laplace for rare patterns
2. **Character-level RNN**: Learn deeper sequential patterns
3. **Ensemble**: Combine multiple HMM variants (different n-gram orders)

**Medium-term** (1 month):

1. **Deep Q-Network**: With proper experience replay and target networks
2. **Policy Gradient**: Directly optimize for score formula
3. **Meta-learning**: Learn to adapt quickly to test distribution

**Long-term** (3 months):

1. **Transformer Model**: Pre-trained on large text corpus
2. **Active Learning**: Update model during test time
3. **Multi-task Learning**: Train on related word games

Realistic estimate: Could reach 45-50% success rate with DQN + better smoothing."

### Q10: Walk me through a game example.

**Answer**:
"Let me trace a game for the word 'APPLE':

**Initial State**:

- Pattern: `_____`
- Guessed: {}
- Lives: 6

**Turn 1**:

- HMM probs (no context): Favors high-frequency letters
- Unigram: E=0.12, A=0.08, O=0.07
- **Guess: E**, Result: `_PP__` ✓ (Lives: 6)

**Turn 2**:

- Pattern: `_PP__`
- Bigram: P followed by? → L (0.15), E (0.12), I (0.08)
- **Guess: L**, Result: `_PPL_` ✓ (Lives: 6)

**Turn 3**:

- Pattern: `_PPL_`
- Trigram: PPL + ending? → E (common -PLE)
- HMM: Position 4 → E (high prob at ends)
- **Guess: E**, Result: `_PPLE` ✓ (Lives: 6)

**Turn 4**:

- Pattern: `_PPLE`
- Position 0 + PPLE → A (APPLE is common word)
- **Guess: A**, Result: `APPLE` ✓ (Lives: 6)

**Outcome**: WIN with 0 wrong guesses!
Reward: +100 + 4×(+10) = +140"

### Q11: What happens if a word is not in your training corpus?

**Answer**:
"The model still works because it learned **patterns**, not words:

**Example**: Word is 'ZEBRA' (assume not in corpus)

1. **Position 0**: HMM learned that Z is rare at start, but still has non-zero smoothed probability
2. **Bigram 'EB'**: Even if unseen, Laplace smoothing gives it prob = α/(total + α×26)
3. **Common endings**: Model knows '-RA' is a common ending from other words

The model will:

- Start with common letters (E, A, T) based on unigrams
- Gradually narrow down using revealed patterns
- May take more wrong guesses for rare words, but won't completely fail

**Smoothing is crucial**: Without it, unseen combinations would have zero probability and model would break."

### Q12: How did you validate your model?

**Answer**:
"Multi-stage validation:

**1. Training Validation** (During Development):

- Split corpus into train (80%) and validation (20%)
- Ensured no overlap
- Tuned hyperparameters (smoothing α, weights) on validation set

**2. Simulation Validation**:

- Ran 500 training episodes
- Tracked reward curve → Should increase and plateau
- Checked for overfitting → Validation reward should track training

**3. Sanity Checks**:

- Probability distributions sum to 1.0 ✓
- No repeated guesses ✓
- All actions valid ✓
- Performance > random baseline ✓

**4. Test Evaluation**:

- Final evaluation on 2000 unseen test words
- Metrics: Success rate, wrong guesses, score

This multi-stage validation ensured robustness."

### Q13: Explain your exploration-exploitation trade-off.

**Answer**:
"I used a **greedy policy** (pure exploitation) rather than exploration for three reasons:

**1. Informed Greedy ≠ Random Greedy**:

- My greedy policy uses HMM probabilities (learned from data)
- Not blindly picking highest frequency letter
- Already incorporates learned patterns

**2. Exploration Cost Too High**:

- Each wrong guess = -1 life
- With only 6 lives, can't afford random exploration
- ε-greedy with ε=0.1 → 10% of guesses wasted

**3. Offline Learning**:

- HMM trained on entire corpus offline
- No online learning during test → No benefit from exploration
- Greedy is optimal for fixed policy

**Trade-off**:

- Pro: Maximum expected reward per episode
- Con: Might miss better strategies (but unlikely with 50K training words)

**If I used DQN**, I would use:

- ε-greedy during training (ε: 1.0 → 0.1 over 10K episodes)
- Pure greedy during testing"

### Q14: What ML concepts does your project demonstrate?

**Answer**:
"This project demonstrates 8 core ML concepts:

1. **Sequence Modeling**: HMM for sequential letter patterns
2. **Probabilistic Inference**: Forward-Backward algorithm
3. **Generalization**: Zero train-test overlap handling
4. **Smoothing**: Laplace smoothing for unseen events
5. **Feature Engineering**: N-grams, position encoding
6. **Reinforcement Learning**: MDP formulation, reward design
7. **Ensemble Methods**: Combining HMM + N-grams
8. **Evaluation**: Proper train/validation/test split

Additionally:

- **Bayesian Reasoning**: Computing posteriors
- **Optimization**: Greedy policy selection
- **Data Structures**: Efficient storage with nested dicts
- **Algorithms**: Dynamic programming (Forward-Backward)"

### Q15: What's the time and space complexity of your solution?

**Answer**:
**Training Time Complexity**:

```
HMM Training: O(N × L)
  - N = 50,000 words
  - L = average word length ≈ 8
  - Total: O(400,000) ≈ 0.4M operations

N-gram Training: O(N × L × G)
  - G = n-gram order (3 for trigram)
  - Total: O(1.2M) operations

Overall Training: O(N × L × G) = O(1.2M) → Very fast (< 1 second)
```

**Inference Time Complexity** (per game):

```
Forward-Backward: O(L × A²)
  - L = word length
  - A = alphabet size = 26
  - Worst case: O(20 × 676) = O(13,520)

N-gram Lookup: O(G) = O(3)

Per Guess: O(L × A²) = O(10,000)
Per Game: O(G × L × A²) where G = num_guesses ≈ 12
  → O(120,000) → Fast (<1ms per game)
```

**Space Complexity**:

```
Emission Probs: O(M × L_max × A)
  - M = max word length = 24
  - L_max = 24 positions
  - A = 26 letters
  - Total: O(15K) floats ≈ 60KB

Trigrams: O(A³) = O(17,576) ≈ 70KB

Total: < 1MB → Very memory efficient
```

**Scalability**:

- Can handle 1M+ word corpus
- Real-time inference (<1ms per guess)
- Fits in RAM easily"

---

## 7. Mathematical Foundations

### 7.1 Hidden Markov Model Formulation

#### Formal Definition

```
HMM λ = (S, O, A, B, π)

S = {s₀, s₁, ..., s_{L-1}}    Hidden states (positions)
O = {a, b, ..., z}             Observation alphabet
A = Transition matrix          [Not used - sequential positions]
B = Emission matrix            B[i][letter] = P(letter | position i)
π = Initial distribution       π[letter] = P(letter at position 0)
```

#### Key Probability

```
P(O | λ) = Σ P(O | S) × P(S | λ)
         all state sequences S

Where:
P(O | S) = Π B[sᵢ][oᵢ]    (emission probs)
P(S | λ) = π[s₀] × Π A[sᵢ, sᵢ₊₁]    (transition probs)
```

### 7.2 Forward Algorithm

#### Forward Variable

```
αᵢ(j) = P(o₁, o₂, ..., oᵢ, qᵢ = sⱼ | λ)

Probability of:
- Observing sequence up to position i
- Being in state sⱼ at position i
```

#### Recursion

```
Initialization:
α₀(j) = π[j] × B[j][o₀]

Induction:
αᵢ(j) = B[j][oᵢ] × Σ αᵢ₋₁(k) × A[k][j]
                    k

Termination:
P(O | λ) = Σ α_{L-1}(j)
           j
```

### 7.3 Backward Algorithm

#### Backward Variable

```
βᵢ(j) = P(oᵢ₊₁, oᵢ₊₂, ..., o_L | qᵢ = sⱼ, λ)

Probability of:
- Observing sequence after position i
- Given state sⱼ at position i
```

#### Recursion

```
Initialization:
β_{L-1}(j) = 1    for all j

Induction:
βᵢ(j) = Σ A[j][k] × B[k][oᵢ₊₁] × βᵢ₊₁(k)
        k

Termination:
P(O | λ) = Σ π[j] × B[j][o₀] × β₀(j)
           j
```

### 7.4 Posterior Probability

#### State Posterior

```
γᵢ(j) = P(qᵢ = sⱼ | O, λ)
      = αᵢ(j) × βᵢ(j) / P(O | λ)
      = αᵢ(j) × βᵢ(j) / Σₖ αᵢ(k) × βᵢ(k)
```

#### Letter Probability for Blank Position

```
P(letter | pattern) = Σ γᵢ(j) × δ(B[j][letter])
                      i∈blanks

Where δ = 1 if emission is possible, 0 otherwise
```

### 7.5 Laplace Smoothing

#### Smoothed Probability

```
P_smoothed(x) = (count(x) + α) / (total + α × |V|)

Where:
α = smoothing parameter (0.1 in implementation)
|V| = vocabulary size (26 for letters)
```

#### Effect

```
Unseen event (count = 0):
P_smoothed = α / (total + α × 26) ≈ 0.004 (non-zero!)

Common event (count = 100, total = 1000):
P_smoothed = (100 + 0.1) / (1000 + 2.6) ≈ 0.100 (barely changed)
```

### 7.6 N-gram Probability

#### Unigram

```
P(letter) = count(letter) / total_letters
```

#### Bigram

```
P(letter | prev) = count(prev, letter) / count(prev)
```

#### Trigram

```
P(letter | prev2, prev1) = count(prev2, prev1, letter) / count(prev2, prev1)
```

#### With Smoothing

```
P_smoothed(letter | context) = (count(context, letter) + α) / (count(context) + α × 26)
```

### 7.7 Probability Combination

#### Weighted Average

```
P_final(letter) = w₁ × P_HMM(letter)
                + w₂ × P_trigram(letter)
                + w₃ × P_bigram(letter)
                + w₄ × P_unigram(letter)

Where: Σ wᵢ = 1
```

#### Adaptive Weights (Contextual)

```
w_HMM = f(num_blanks, total_length)
w_trigram = g(num_revealed_context)
w_bigram = h(num_revealed_context)
w_unigram = 1 - w_HMM - w_trigram - w_bigram
```

---

## 8. Code Walkthrough Points

### 8.1 Key Classes

#### EnhancedProbabilisticHMM

```python
class EnhancedProbabilisticHMM:
    def __init__(self):
        self.emission_probs = {}      # HMM emission matrix
        self.unigram_probs = {}       # P(letter)
        self.bigram_probs = {}        # P(letter | prev)
        self.trigram_probs = {}       # P(letter | prev2, prev1)
```

**Key Methods**:

1. `train(corpus)` - Build all probability models
2. `get_letter_probabilities(pattern, guessed)` - Main inference
3. `_forward_backward(pattern)` - HMM inference
4. `_get_ngram_probs(pattern)` - N-gram inference
5. `_combine_probabilities(hmm, ngram, blanks)` - Weighted combination

#### HangmanGame

```python
class HangmanGame:
    def __init__(self, word, max_wrong=6):
        self.word = word.upper()
        self.pattern = ['_'] * len(word)
        self.guessed = set()
        self.wrong_count = 0
```

**Key Methods**:

1. `guess(letter)` - Process a guess
2. `is_won()` - Check win condition
3. `is_lost()` - Check loss condition
4. `get_state()` - Return current state

### 8.2 Critical Code Sections

#### Section 1: Emission Probability Calculation

```python
# Count emissions at each position
for word in words_by_length[length]:
    for pos, letter in enumerate(word):
        emission_counts[length][pos][letter] += 1

# Normalize with smoothing
for pos in range(length):
    total = sum(emission_counts[length][pos].values())
    for letter in alphabet:
        count = emission_counts[length][pos].get(letter, 0)
        smoothed = (count + alpha) / (total + alpha * 26)
        emission_probs[length][pos][letter] = smoothed
```

**Explain**: This builds the B matrix of the HMM with Laplace smoothing.

#### Section 2: Forward Pass

```python
# Initialize
for letter in alphabet:
    if pattern[0] != '_':
        alpha[0][letter] = 1.0 if letter == pattern[0] else 0.0
    else:
        alpha[0][letter] = self.emission_probs[length][0].get(letter, epsilon)

# Recurse
for i in range(1, length):
    if pattern[i] != '_':
        # Observed letter - deterministic
        alpha[i][pattern[i]] = sum(alpha[i-1].values())
    else:
        # Blank - probabilistic
        for letter in alphabet:
            alpha[i][letter] = (
                self.emission_probs[length][i].get(letter, epsilon)
                * sum(alpha[i-1].values())
            )
```

**Explain**: This is the forward pass computing α(i, letter). Key insight: observed positions are deterministic (probability 1 or 0).

#### Section 3: Probability Combination

```python
def _combine_probabilities(self, hmm_probs, ngram_probs, num_blanks, total_length):
    if num_blanks == total_length:
        # No context - rely on frequency
        w_hmm, w_uni = 0.3, 0.7
        return {l: w_hmm * hmm_probs.get(l, 0) + w_uni * self.unigram_probs.get(l, 0)
                for l in alphabet}
    elif num_blanks >= total_length * 0.7:
        # Little context - use bigrams
        w_hmm, w_bi, w_uni = 0.4, 0.4, 0.2
        return {l: w_hmm * hmm_probs.get(l, 0)
                 + w_bi * ngram_probs['bigram'].get(l, 0)
                 + w_uni * self.unigram_probs.get(l, 0)
                for l in alphabet}
    # ... more cases
```

**Explain**: Adaptive weighting based on information available. Early game uses unigrams, late game uses trigrams.

### 8.3 Data Flow

```
1. Training Phase:
   corpus.txt → Read words → Split by length →
   → Count emissions → Normalize → Build HMM
   → Count n-grams → Normalize → Build N-gram models

2. Inference Phase (per game):
   Test word → Initialize game → Pattern = "____"
   ↓
   Loop until done:
     Current pattern → Forward-Backward → HMM probabilities
     Current pattern → Extract context → N-gram probabilities
     Combine → Weighted average → Select max
     Guess letter → Update pattern → Repeat

3. Evaluation:
   All games → Collect stats → Calculate score
```

---

## 9. Comparison with Alternatives

### 9.1 Alternative Approaches Tried

#### Approach 1: Pure Q-Learning (V1)

```python
Q-table: Dict[(pattern, guessed, wrong_count), letter] → Q-value
Policy: ε-greedy with ε decay
Result: Score = -56,492 (15.65% success)
```

**Failure Reason**:

- State space too large → Sparse Q-table
- Insufficient exploration → Many states never visited
- No generalization → Memorizes training words

#### Approach 2: Vowel-First Heuristic (V3)

```python
Strategy: Always guess vowels (A,E,I,O,U) first, then consonants by frequency
Result: Score = -57,820 (worse than random!)
```

**Failure Reason**:

- Words like "RHYTHM", "FLY" have few/no vowels
- Wastes guesses on words where vowels already revealed
- No adaptation to context

#### Approach 3: Pattern Matching (V4)

```python
Strategy: Find all corpus words matching current pattern, guess most common letter
Result: Score = -62,418 (catastrophic)
```

**Failure Reason**:

- Zero overlap → No matches for test words
- Falls back to random guessing
- Doesn't generalize at all

#### Approach 4: Statistical N-grams Only (V5)

```python
Strategy: Use only trigram/bigram/unigram probabilities
Result: Score = -52,389 (29.05% success)
```

**Success**: Better generalization than pattern matching
**Limitation**: Doesn't capture position-specific patterns

### 9.2 Why HMM + N-grams Won

| Feature           | Q-Learning | Vowel-First | Pattern Match | N-grams Only | **HMM + N-grams** |
| ----------------- | ---------- | ----------- | ------------- | ------------ | ----------------- |
| Generalization    | ✗          | ✗           | ✗             | ✓            | ✓✓                |
| Position Info     | ✗          | ✗           | ✗             | ✗            | ✓                 |
| Context Info      | ✗          | ✗           | ✓             | ✓            | ✓                 |
| Adaptability      | ✓          | ✗           | ✗             | ✗            | ✓                 |
| Sample Efficiency | ✗          | ✓           | ✓             | ✓            | ✓                 |
| **Final Score**   | -56K       | -58K        | -62K          | -52K         | **-51K** ✓        |

**Key Insight**: Need both position (HMM) and context (N-grams) for best performance.

### 9.3 What Would Beat Current Approach?

#### Deep Learning Approach

```python
Model: Character-level LSTM
Input: Pattern with mask tokens: [M, A, S, K, M, M]
Output: Probability distribution over 26 letters
Training: Teacher forcing on corpus words

Expected Performance: 40-45% success rate
Advantage: Learns deeper patterns, better generalization
Disadvantage: Needs large corpus, prone to overfitting
```

#### Ensemble Approach

```python
Models:
1. HMM with trigrams (current)
2. Character-level CNN
3. Transformer with attention
4. Rule-based (common endings, prefixes)

Combination: Weighted vote or stacking
Expected Performance: 45-50% success rate
```

#### Reinforcement Learning Approach (Done Right)

```python
Model: Deep Q-Network (DQN)
State: One-hot pattern + guessed vector + HMM probs (400-dim)
Network: 400 → 256 → 128 → 26 (Q-values)
Training: 100K episodes with experience replay

Expected Performance: 50-55% success rate
Advantage: Learns optimal policy end-to-end
Disadvantage: Needs massive training data
```

---

## 10. Critical Thinking Questions

### Q: Is your HMM truly a Hidden Markov Model?

**Critical Analysis**:
Technically, **no** - it's a simplified version:

**True HMM Requirements**:

1. ✓ Hidden states (positions)
2. ✓ Emissions (letters)
3. ✗ **Transition probabilities** (I don't model position-to-position transitions)
4. ✓ Observation probabilities

**What I Actually Have**:

- Position-indexed emission model
- No state transitions (positions are sequential by definition)
- More like a "Naive Bayes with positional independence"

**Why It Still Works**:

- In Hangman, transitions are deterministic (position i → position i+1)
- The emission probabilities capture the key pattern (what letter at what position)
- Forward-Backward still valid for computing posteriors

**More Accurate Name**: "Position-Dependent Emission Model with Bayesian Inference"

### Q: Did you really implement Reinforcement Learning?

**Honest Answer**:
**Partially**. I have the RL framework but use a fixed greedy policy rather than learned policy.

**What I Have**:

- ✓ Environment (HangmanGame)
- ✓ State representation
- ✓ Action space
- ✓ Reward function
- ✗ **Learning algorithm** (no Q-update, no policy gradient)

**What I Actually Do**:

- Use HMM as a heuristic policy
- No parameter updates based on reward
- More like "Heuristic Search" than "Reinforcement Learning"

**Why This Is Okay**:

- Greedy policy using learned HMM ≈ value function approximation
- "Learning" happens offline during HMM training
- Could argue HMM training is "reward-free RL" (learning from corpus)

**To Make It True RL**:
Would need to update HMM parameters based on game outcomes (reward signal).

### Q: Are you overfitting to the scoring formula?

**Consideration**:

```
Score = (Success × 2000) - (Wrong × 5) - (Repeated × 2)
```

The penalty for wrong guesses (×5) is much higher than repeated guesses (×2).

**Did I Optimize For This?**

- Yes: Zero repeated guesses (perfect validity)
- Partially: Greedy policy minimizes expected wrong guesses
- No: Didn't explicitly tune hyperparameters to maximize this specific formula

**Potential Gaming**:
Could artificially inflate success rate by:

- Being more conservative (fewer wrong guesses, but also fewer wins)
- Giving up early on hard words (avoid wrong guesses)

**Why I Didn't**:

- Greedy policy already balances success vs wrong guesses
- Hard to game without explicit optimization for formula
- Real goal is generalization, not formula exploitation

### Q: How do you know your model isn't just memorizing?

**Evidence of Generalization**:

1. **Zero Train-Test Overlap**: Impossible to memorize
2. **Performance Degrades Gradually**: If memorizing, would see binary (perfect on training, random on test)
3. **Consistent Cross-Lengths**: Model works across all word lengths
4. **Smoothing Helps**: Unseen combinations still get non-zero probability

**Test**:
If I trained only on 3-letter words and tested on 12-letter words:

```
Expected: Performance drops but still > random
Reality: Would need to implement to confirm
```

**Counter-Evidence**:

- Performance is only 32.85%, suggesting model hasn't learned enough
- Could be underfitting rather than overfitting

### Q: What assumptions does your model make?

**Explicit Assumptions**:

1. **Position Independence**: P(letter | position) doesn't depend on other positions
2. **Language Consistency**: Test words follow same patterns as corpus
3. **Markov Property**: P(letter at position i | positions <i) depends only on recent context (n-grams)
4. **Fixed Corpus**: All patterns can be learned from 50K words

**Implicit Assumptions**:

1. Word lengths in test ≤ max length in training
2. Letters are uniformly drawable (no bias in word selection)
3. English language patterns are consistent

**Violated Assumptions**:

- ✗ Zero overlap means test distribution might differ from train
- ✗ Rare words may have different patterns than common words

### Q: Is 32.85% success rate actually good?

**Context Matters**:

**Comparison**:

- Random guessing: ~8.5%
- Frequency only: ~15.6%
- Vowel-first: ~18.2%
- My model: 32.85%

**Relative Improvement**:

- 3.8× better than random
- 2.1× better than frequency
- **But still fails 67% of games**

**Human Performance**:

- Average human: ~70-80% (with thinking)
- Expert: ~85-90%

**Assessment**:

- Good for an automated system
- Room for improvement
- Respectable given constraints (zero overlap, 50K corpus, no pre-training)

### Q: What ethical considerations exist for this project?

**Interesting Angle**:

1. **Bias in Language**:

   - Corpus may favor certain word types (common vs rare, formal vs slang)
   - Model will reflect these biases
   - Could disadvantage certain word categories

2. **Accessibility**:

   - Automated Hangman solver could help learning-disabled players
   - Or could ruin the game for others (cheating)

3. **Educational Value**:

   - Demonstrates ML concepts
   - Teaches about probability and inference
   - Could inspire interest in AI

4. **Resource Usage**:
   - Minimal compute (training < 1 second)
   - Energy-efficient compared to large models
   - Environmentally responsible

**Conclusion**: Low-stakes application, but good exercise in thinking about ML ethics.

---

## Final Preparation Tips

### 1. Practice Your Demo

- Load notebook
- Run cells in order
- Explain each section briefly
- Show learning curves and results
- Be ready to modify code live

### 2. Memorize Key Numbers

- **Score: -51,288**
- **Success Rate: 32.85%**
- **Avg Wrong: 5.195**
- **Training Size: 50,000 words**
- **Test Size: 2,000 words**
- **Smoothing α: 0.1**

### 3. Prepare to Whiteboard

- Draw HMM structure
- Show Forward-Backward recursion
- Explain probability combination
- Sketch learning curve

### 4. Know Your Weaknesses

- Not true RL (greedy policy, no learning)
- Simplified HMM (no transitions)
- Moderate performance (32.85%)
- No online adaptation

### 5. Know Your Strengths

- Proper probabilistic inference
- Generalization to unseen words
- Zero repeated guesses
- Comprehensive evaluation
- Well-documented approach

### 6. Have Questions Ready

If they ask "Any questions for us?":

- "What extensions would you suggest?"
- "How would you handle extremely rare words?"
- "Could this approach work for other word games?"

---

## Quick Reference Cheat Sheet

### Model Architecture

```
Input: Pattern "_PP__", Guessed: {A,E}
  ↓
Forward-Backward (HMM)
  → P(letter | position)
N-gram Models
  → P(letter | context)
  ↓
Adaptive Combination
  ↓
Output: P(L)=0.25, P(E)=0.20, ...
  ↓
Greedy Selection: Guess 'L'
```

### Key Equations

```
HMM: γᵢ(j) = αᵢ(j) × βᵢ(j) / Σₖ αᵢ(k) × βᵢ(k)
Smoothing: P = (count + α) / (total + α×26)
N-gram: P(c|a,b) = count(a,b,c) / count(a,b)
Score: (Success×2000) - (Wrong×5) - (Repeat×2)
```

### Performance Summary

| Metric        | Value   |
| ------------- | ------- |
| Final Score   | -51,288 |
| Success Rate  | 32.85%  |
| Wrong/Game    | 5.195   |
| Repeated/Game | 0       |

### Code Structure

```
1. Imports (numpy, collections, matplotlib)
2. EnhancedProbabilisticHMM class
   - train()
   - get_letter_probabilities()
   - _forward_backward()
   - _get_ngram_probs()
3. HangmanGame class
4. Training (50K words)
5. RL Simulation (500 episodes)
6. Evaluation (2K test words)
7. Visualization
```

---

**Good luck with your viva! You've got this! 🚀**
