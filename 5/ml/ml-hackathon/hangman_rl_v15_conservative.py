"""
Hangman Agent - Conservative RL with HMM Backbone
Version 15: True RL with Minimal Exploration on V8 HMM

Combining:
1. HMM Forward-Backward for position-based probabilities (from V8)
2. Conditional probabilities from bigrams/trigrams (from V8)
3. Epsilon-greedy exploration (2% exploration rate)
4. Q-value adjustments based on experience
5. Target: Maintain 30%+ accuracy while having TRUE RL components
"""

import numpy as np
from collections import defaultdict, Counter
from typing import List, Tuple, Set, Dict
import pickle
import random


class EnhancedProbabilisticHMM:
    """
    Enhanced HMM with n-gram conditional probabilities and RL components.
    """
    
    def __init__(self, max_length: int = 24, epsilon: float = 0.02, learning_rate: float = 0.01):
        self.max_length = max_length
        self.alphabet = 'abcdefghijklmnopqrstuvwxyz'
        self.n_letters = len(self.alphabet)
        self.letter_to_idx = {letter: i for i, letter in enumerate(self.alphabet)}
        self.idx_to_letter = {i: letter for i, letter in enumerate(self.alphabet)}
        
        # HMM parameters by length
        self.hmms = {}
        
        # N-gram probabilities (for contextual information)
        self.bigram_probs = defaultdict(lambda: defaultdict(float))  # P(letter2 | letter1)
        self.trigram_probs = defaultdict(lambda: defaultdict(float))  # P(letter3 | letter1, letter2)
        self.letter_probs = defaultdict(float)  # P(letter)
        
        # RL components
        self.epsilon = epsilon  # Exploration rate
        self.learning_rate = learning_rate
        self.q_adjustments = defaultdict(float)  # Small adjustments to probabilities
        self.state_visits = defaultdict(int)  # Track state visits
        
        # Training statistics
        self.training_rewards = []
        self.training_wins = []
        
    def train(self, corpus_file: str):
        """Train HMM and n-gram models."""
        print("Training Enhanced Probabilistic HMM...")
        
        with open(corpus_file, 'r') as f:
            words = [line.strip().lower() for line in f if line.strip()]
        
        # Group by length for HMM
        words_by_length = defaultdict(list)
        for word in words:
            if len(word) <= self.max_length:
                words_by_length[len(word)].append(word)
        
        print(f"Training on {len(words)} words")
        
        # Train position-based HMM for each length
        for length in sorted(words_by_length.keys()):
            if len(words_by_length[length]) >= 5:
                self.hmms[length] = self._train_hmm(words_by_length[length], length)
        
        # Train n-gram models (pooled across all lengths)
        self._train_ngrams(words)
        
        print(f"Trained HMMs for lengths: {sorted(self.hmms.keys())}")
        print(f"Top unigrams: {sorted(self.letter_probs.items(), key=lambda x: x[1], reverse=True)[:10]}")
    
    def _train_hmm(self, words: List[str], length: int) -> Dict:
        """Train HMM for specific length."""
        # Emission probabilities B[pos, letter]
        emission_counts = np.zeros((length, self.n_letters))
        
        for word in words:
            for pos, letter in enumerate(word):
                if letter in self.letter_to_idx:
                    letter_idx = self.letter_to_idx[letter]
                    emission_counts[pos, letter_idx] += 1
        
        # Normalize with smoothing
        B = np.zeros((length, self.n_letters))
        alpha = 0.5  # Smoothing
        for pos in range(length):
            smoothed = emission_counts[pos] + alpha
            B[pos] = smoothed / smoothed.sum()
        
        return {'length': length, 'B': B, 'n_words': len(words)}
    
    def _train_ngrams(self, words: List[str]):
        """Train n-gram probability models."""
        # Count frequencies
        unigram_counts = Counter()
        bigram_counts = defaultdict(Counter)
        trigram_counts = defaultdict(Counter)
        
        for word in words:
            for letter in word:
                unigram_counts[letter] += 1
            
            for i in range(len(word) - 1):
                bigram_counts[word[i]][word[i+1]] += 1
            
            for i in range(len(word) - 2):
                trigram_counts[word[i:i+2]][word[i+2]] += 1
        
        # Convert to probabilities
        total = sum(unigram_counts.values())
        for letter, count in unigram_counts.items():
            self.letter_probs[letter] = count / total
        
        # Bigram probabilities with smoothing
        for letter1 in self.alphabet:
            total_count = sum(bigram_counts[letter1].values())
            if total_count > 0:
                for letter2 in self.alphabet:
                    count = bigram_counts[letter1].get(letter2, 0)
                    # Laplace smoothing
                    self.bigram_probs[letter1][letter2] = (count + 0.1) / (total_count + 0.1 * self.n_letters)
        
        # Trigram probabilities
        for bigram, next_counts in trigram_counts.items():
            total_count = sum(next_counts.values())
            if total_count > 0:
                for letter3 in self.alphabet:
                    count = next_counts.get(letter3, 0)
                    self.trigram_probs[bigram][letter3] = (count + 0.1) / (total_count + 0.1 * self.n_letters)
    
    def get_letter_probabilities(self, pattern: str, guessed: Set[str]) -> Dict[str, float]:
        """
        Combine multiple probability sources:
        1. Position-based HMM emissions
        2. Bigram conditional probabilities
        3. Trigram conditional probabilities
        4. Unigram fallback
        """
        length = len(pattern)
        remaining = set(self.alphabet) - guessed
        
        if not remaining:
            return {}
        
        letter_scores = defaultdict(float)
        
        # Source 1: Position-based HMM probabilities
        if length in self.hmms:
            hmm = self.hmms[length]
            B = hmm['B']
            
            unknown_positions = [i for i, c in enumerate(pattern) if c == '_']
            for pos in unknown_positions:
                for letter in remaining:
                    letter_idx = self.letter_to_idx[letter]
                    # Weight by emission probability
                    letter_scores[letter] += B[pos, letter_idx] * 3.0  # Higher weight for position-based
        
        # Source 2: Bigram conditional probabilities
        for i, char in enumerate(pattern):
            if char != '_':
                # Look right: P(next | current)
                if i + 1 < length and pattern[i + 1] == '_':
                    for letter in remaining:
                        if letter in self.bigram_probs[char]:
                            letter_scores[letter] += self.bigram_probs[char][letter] * 2.0
                
                # Look left: P(current | prev)
                if i - 1 >= 0 and pattern[i - 1] == '_':
                    for letter in remaining:
                        if char in self.bigram_probs.get(letter, {}):
                            letter_scores[letter] += self.bigram_probs[letter][char] * 2.0
        
        # Source 3: Trigram conditional probabilities
        for i in range(len(pattern) - 2):
            # Pattern: known, known, unknown
            if pattern[i] != '_' and pattern[i+1] != '_' and pattern[i+2] == '_':
                bigram = pattern[i:i+2]
                if bigram in self.trigram_probs:
                    for letter in remaining:
                        letter_scores[letter] += self.trigram_probs[bigram].get(letter, 0) * 2.5
            
            # Pattern: known, unknown, known
            if pattern[i] != '_' and pattern[i+1] == '_' and pattern[i+2] != '_':
                # This is trickier - we need P(middle | first, third)
                # Use approximation: P(middle | first) * P(third | middle)
                for letter in remaining:
                    if letter in self.bigram_probs[pattern[i]]:
                        prob1 = self.bigram_probs[pattern[i]][letter]
                        prob2 = self.bigram_probs.get(letter, {}).get(pattern[i+2], 0)
                        letter_scores[letter] += (prob1 * prob2) * 1.5
            
            # Pattern: unknown, known, known
            if pattern[i] == '_' and pattern[i+1] != '_' and pattern[i+2] != '_':
                bigram = pattern[i+1:i+3]
                # Find letters that could precede this bigram
                for letter in remaining:
                    test_trigram = letter + bigram
                    if bigram in self.trigram_probs:
                        # Check if letter appears before bigram
                        for prev_letter in self.alphabet:
                            if prev_letter == letter:
                                bigram_key = prev_letter + pattern[i+1]
                                if bigram_key in self.trigram_probs:
                                    letter_scores[letter] += self.trigram_probs[bigram_key].get(pattern[i+2], 0) * 1.5
        
        # Source 4: Unigram fallback (always add base probability)
        for letter in remaining:
            letter_scores[letter] += self.letter_probs.get(letter, 1e-5) * 1.0
        
        # Normalize
        total = sum(letter_scores.values())
        if total > 0:
            return {letter: score / total for letter, score in letter_scores.items()}
        
        # Ultimate fallback
        return {letter: 1.0 / len(remaining) for letter in remaining}
    
    def choose_letter(self, pattern: str, guessed: Set[str], training: bool = False) -> str:
        """
        Choose letter with highest probability, with epsilon-greedy exploration if training.
        """
        probs = self.get_letter_probabilities(pattern, guessed)
        
        if not probs:
            for letter in 'etaoinshrdlcumwfgypbvkjxqz':
                if letter not in guessed:
                    return letter
            return 'e'
        
        # Epsilon-greedy exploration during training
        if training and random.random() < self.epsilon:
            # Explore: weighted random based on probabilities
            letters = list(probs.keys())
            weights = [probs[l] for l in letters]
            return random.choices(letters, weights=weights)[0]
        
        # Exploit: choose best action with Q-adjustments
        state_key = self._get_state_key(pattern, guessed)
        adjusted_probs = {}
        
        for letter, prob in probs.items():
            q_key = (state_key, letter)
            q_adj = self.q_adjustments.get(q_key, 0.0)
            # Apply small adjustment (clamped to not overwhelm HMM probs)
            adjusted_probs[letter] = prob + min(0.1, max(-0.1, q_adj))
        
        return max(adjusted_probs.items(), key=lambda x: x[1])[0]
    
    def _get_state_key(self, pattern: str, guessed: Set[str]) -> str:
        """Get simplified state representation for Q-table."""
        num_blanks = pattern.count('_')
        num_guessed = len(guessed)
        return f"{len(pattern)}|{num_blanks}|{num_guessed}"
    
    def update_q_values(self, pattern: str, guessed: Set[str], action: str, reward: float):
        """Update Q-value adjustments based on reward (RL component)."""
        state_key = self._get_state_key(pattern, guessed)
        q_key = (state_key, action)
        
        self.state_visits[q_key] += 1
        
        # Simple Q-learning-style update with decaying learning rate
        alpha = self.learning_rate / (1 + self.state_visits[q_key] * 0.01)
        current_q = self.q_adjustments.get(q_key, 0.0)
        
        # Update based on reward signal
        self.q_adjustments[q_key] = current_q + alpha * reward


class HangmanGame:
    """Hangman game."""
    
    def __init__(self, word: str, max_wrong: int = 6):
        self.word = word.lower()
        self.max_wrong = max_wrong
        self.reset()
    
    def reset(self):
        self.pattern = '_' * len(self.word)
        self.guessed = set()
        self.wrong = 0
        self.repeated = 0
    
    def guess(self, letter: str):
        letter = letter.lower()
        
        if letter in self.guessed:
            self.repeated += 1
            return
        
        self.guessed.add(letter)
        
        if letter in self.word:
            new_pattern = []
            for i, c in enumerate(self.word):
                if c == letter:
                    new_pattern.append(letter)
                else:
                    new_pattern.append(self.pattern[i])
            self.pattern = ''.join(new_pattern)
        else:
            self.wrong += 1
    
    def is_won(self):
        return '_' not in self.pattern
    
    def is_lost(self):
        return self.wrong >= self.max_wrong
    
    def is_done(self):
        return self.is_won() or self.is_lost()


def play_game_with_training(hmm: EnhancedProbabilisticHMM, word: str, training: bool = False) -> Tuple[bool, int, int, float]:
    """Play one game with optional RL training."""
    game = HangmanGame(word)
    total_reward = 0
    episode_transitions = []
    
    while not game.is_done():
        pattern = game.pattern
        guessed = game.guessed.copy()
        
        letter = hmm.choose_letter(pattern, guessed, training=training)
        
        prev_wrong = game.wrong
        prev_pattern = game.pattern
        game.guess(letter)
        
        # Calculate reward for this action
        if letter in word:
            # Correct guess
            num_revealed = game.pattern.count(letter) - prev_pattern.count(letter)
            reward = 10 + (num_revealed * 2)
        else:
            # Wrong guess
            reward = -15
        
        total_reward += reward
        
        if training:
            episode_transitions.append((pattern, guessed, letter, reward))
    
    # Terminal reward
    if game.is_won():
        total_reward += 100
        terminal_reward = 50
    else:
        total_reward -= 50
        terminal_reward = -30
    
    # Update Q-values if training
    if training and episode_transitions:
        # Update all transitions with discounted rewards
        for i, (pattern, guessed, action, reward) in enumerate(episode_transitions):
            # Add terminal reward to last action
            if i == len(episode_transitions) - 1:
                reward += terminal_reward
            
            hmm.update_q_values(pattern, guessed, action, reward)
    
    return (game.is_won(), game.wrong, game.repeated, total_reward)


def train_with_rl(hmm: EnhancedProbabilisticHMM, train_words: List[str], num_episodes: int = 3000):
    """Train HMM with RL fine-tuning."""
    print(f"\nRL Fine-Tuning for {num_episodes} episodes...")
    
    for episode in range(num_episodes):
        word = random.choice(train_words)
        won, wrong, repeated, reward = play_game_with_training(hmm, word, training=True)
        
        hmm.training_rewards.append(reward)
        hmm.training_wins.append(1 if won else 0)
        
        if (episode + 1) % 500 == 0:
            recent_wins = sum(hmm.training_wins[-500:])
            avg_reward = np.mean(hmm.training_rewards[-500:])
            print(f"  Episode {episode + 1}/{num_episodes} | "
                  f"Win Rate: {recent_wins/500:.2%} | "
                  f"Avg Reward: {avg_reward:.2f} | "
                  f"Q-table size: {len(hmm.q_adjustments)}")
    
    print("RL fine-tuning completed!")


def play_game(hmm: EnhancedProbabilisticHMM, word: str) -> Tuple[bool, int, int]:
    """Play one game without training."""
    won, wrong, repeated, _ = play_game_with_training(hmm, word, training=False)
    return (won, wrong, repeated)


def evaluate(hmm: EnhancedProbabilisticHMM, test_file: str):
    """Evaluate on test set."""
    print("\nEvaluating...")
    
    with open(test_file, 'r') as f:
        test_words = [line.strip().lower() for line in f if line.strip()]
    
    wins = 0
    total_wrong = 0
    total_repeated = 0
    
    for i, word in enumerate(test_words):
        won, wrong, repeated = play_game(hmm, word)
        
        wins += won
        total_wrong += wrong
        total_repeated += repeated
        
        if (i + 1) % 500 == 0:
            success_rate = wins / (i + 1)
            print(f"  Progress: {i+1}/{len(test_words)}, Success: {success_rate*100:.2f}%")
    
    n = len(test_words)
    success_rate = wins / n
    avg_wrong = total_wrong / n
    avg_repeated = total_repeated / n
    
    final_score = (success_rate * 2000) - (total_wrong * 5) - (total_repeated * 2)
    
    print("\n" + "=" * 70)
    print("RESULTS - VERSION 8 (Enhanced HMM + N-grams)")
    print("=" * 70)
    print(f"Success Rate: {success_rate*100:.2f}% ({wins}/{n})")
    print(f"Average Wrong Guesses: {avg_wrong:.2f}")
    print(f"Average Repeated Guesses: {avg_repeated:.2f}")
    print(f"Total Wrong Guesses: {total_wrong}")
    print(f"Total Repeated Guesses: {total_repeated}")
    print(f"\nFINAL SCORE: {final_score:.2f}")
    print("=" * 70)
    print("\nScore Breakdown:")
    print(f"  Success bonus:     +{success_rate * 2000:.2f}")
    print(f"  Wrong penalty:     -{total_wrong * 5:.2f}")
    print(f"  Repeated penalty:  -{total_repeated * 2:.2f}")
    print("=" * 70)
    
    return final_score


def main():
    print("=" * 70)
    print("Hangman Agent V15 - Conservative RL with HMM Backbone")
    print("True RL: Epsilon-Greedy Exploration + Q-Learning Updates")
    print("=" * 70)
    
    corpus_file = 'data/corpus.txt'
    test_file = 'data/test.txt'
    
    # Load training data
    with open(corpus_file, 'r') as f:
        train_words = [line.strip().lower() for line in f if line.strip()]
    
    # Train HMM
    hmm = EnhancedProbabilisticHMM(max_length=24, epsilon=0.02, learning_rate=0.01)
    hmm.train(corpus_file)
    
    # RL fine-tuning
    train_with_rl(hmm, train_words, num_episodes=3000)
    
    # Evaluate
    score = evaluate(hmm, test_file)
    
    # Save
    with open('models/hmm_rl_model_v15.pkl', 'wb') as f:
        pickle.dump(hmm, f)
    print("\nModel saved to models/hmm_rl_model_v15.pkl")
    
    return score


if __name__ == "__main__":
    main()
