"""
Hangman Agent - Proper Probabilistic HMM
Version 7: Enhanced Forward-Backward with Proper Inference

True probabilistic approach:
- Train HMM parameters (transitions, emissions) from corpus
- Use Forward-Backward to compute marginal probabilities
- Infer P(letter | observed pattern) probabilistically
- No statistical counting - pure probability inference
"""

import numpy as np
from collections import defaultdict
from typing import List, Tuple, Set, Dict
import pickle


class ProbabilisticHMM:
    """
    Proper HMM with Forward-Backward inference.
    
    For Hangman:
    - States: Positions in word (0, 1, ..., L-1)
    - Observations: Letters (a-z) or unknown (_)
    - Use Forward-Backward to compute P(letter at position | partial observation)
    """
    
    def __init__(self, max_length: int = 24):
        self.max_length = max_length
        self.alphabet = 'abcdefghijklmnopqrstuvwxyz'
        self.n_letters = len(self.alphabet)
        self.letter_to_idx = {letter: i for i, letter in enumerate(self.alphabet)}
        self.idx_to_letter = {i: letter for i, letter in enumerate(self.alphabet)}
        
        # Store HMM parameters by word length
        self.hmms = {}
        
    def train(self, corpus_file: str):
        """Train HMM parameters from corpus."""
        print("Training Probabilistic HMM...")
        
        with open(corpus_file, 'r') as f:
            words = [line.strip().lower() for line in f if line.strip()]
        
        # Group by length
        words_by_length = defaultdict(list)
        for word in words:
            if len(word) <= self.max_length:
                words_by_length[len(word)].append(word)
        
        print(f"Training on {len(words)} words")
        
        # Train HMM for each length
        for length in sorted(words_by_length.keys()):
            if len(words_by_length[length]) >= 5:
                self.hmms[length] = self._estimate_parameters(words_by_length[length], length)
        
        print(f"Trained HMMs for lengths: {sorted(self.hmms.keys())}")
    
    def _estimate_parameters(self, words: List[str], length: int) -> Dict:
        """
        Estimate HMM parameters using Maximum Likelihood Estimation.
        
        Returns dict with:
        - pi: Initial state distribution
        - A: Transition matrix
        - B: Emission matrix
        """
        n_words = len(words)
        
        # Initial state probabilities (always start at position 0)
        pi = np.zeros(length)
        pi[0] = 1.0
        
        # Transition probabilities (deterministic left-to-right)
        A = np.zeros((length, length))
        for i in range(length - 1):
            A[i, i + 1] = 1.0
        A[length - 1, length - 1] = 1.0
        
        # Emission probabilities B[state, letter]
        # B[i, j] = P(letter_j | position_i)
        emission_counts = np.zeros((length, self.n_letters))
        
        for word in words:
            for pos, letter in enumerate(word):
                if letter in self.letter_to_idx:
                    letter_idx = self.letter_to_idx[letter]
                    emission_counts[pos, letter_idx] += 1
        
        # Normalize with Laplace smoothing
        B = np.zeros((length, self.n_letters))
        alpha = 0.1  # Smoothing parameter
        for pos in range(length):
            smoothed = emission_counts[pos] + alpha
            B[pos] = smoothed / smoothed.sum()
        
        return {
            'length': length,
            'pi': pi,
            'A': A,
            'B': B,
            'n_words': n_words
        }
    
    def forward(self, hmm: Dict, pattern: str, guessed: Set[str]) -> np.ndarray:
        """
        Forward algorithm: compute alpha[t, i] = P(obs_1:t, state_t=i | model)
        
        For Hangman, observations are:
        - Known letters at certain positions
        - Unknown positions (could be any letter not yet guessed)
        """
        length = hmm['length']
        pi = hmm['pi']
        A = hmm['A']
        B = hmm['B']
        
        alpha = np.zeros((length, length))
        
        # Initialization (t=0)
        obs_prob = self._observation_probability(B, 0, pattern[0], guessed)
        alpha[0, :] = pi * obs_prob
        
        # Normalize to prevent underflow
        alpha[0, :] /= (alpha[0, :].sum() + 1e-10)
        
        # Recursion
        for t in range(1, length):
            obs_prob = self._observation_probability(B, t, pattern[t], guessed)
            
            for j in range(length):
                # Sum over all previous states
                alpha[t, j] = np.sum(alpha[t-1, :] * A[:, j]) * obs_prob[j]
            
            # Normalize
            alpha[t, :] /= (alpha[t, :].sum() + 1e-10)
        
        return alpha
    
    def backward(self, hmm: Dict, pattern: str, guessed: Set[str]) -> np.ndarray:
        """
        Backward algorithm: compute beta[t, i] = P(obs_t+1:T | state_t=i, model)
        """
        length = hmm['length']
        A = hmm['A']
        B = hmm['B']
        
        beta = np.zeros((length, length))
        
        # Initialization (t=T)
        beta[length - 1, :] = 1.0
        
        # Recursion (backward)
        for t in range(length - 2, -1, -1):
            obs_prob = self._observation_probability(B, t + 1, pattern[t + 1], guessed)
            
            for i in range(length):
                # Sum over all next states
                beta[t, i] = np.sum(A[i, :] * obs_prob * beta[t + 1, :])
            
            # Normalize
            beta[t, :] /= (beta[t, :].sum() + 1e-10)
        
        return beta
    
    def _observation_probability(self, B: np.ndarray, pos: int, obs: str, guessed: Set[str]) -> np.ndarray:
        """
        Compute observation probability at position.
        
        Returns: array of probabilities for each state (position)
        In our model, state = position, so we just return prob for this position.
        """
        if obs != '_':
            # Known letter - use emission probability directly
            if obs in self.letter_to_idx:
                letter_idx = self.letter_to_idx[obs]
                prob = np.zeros(len(B))
                prob[pos] = B[pos, letter_idx]
                return prob
            else:
                return np.zeros(len(B))
        else:
            # Unknown position - could be any unguessed letter
            # P(obs=unknown | state) = sum of P(letter | state) for all unguessed letters
            prob = np.zeros(len(B))
            remaining_prob = 0.0
            
            for letter in self.alphabet:
                if letter not in guessed:
                    letter_idx = self.letter_to_idx[letter]
                    remaining_prob += B[pos, letter_idx]
            
            prob[pos] = remaining_prob if remaining_prob > 0 else 1e-10
            return prob
    
    def get_letter_probabilities(self, pattern: str, guessed: Set[str]) -> Dict[str, float]:
        """
        Use Forward-Backward to compute P(letter | observed pattern).
        
        For each unknown position, compute:
        P(letter at position | pattern) using smoothing over alpha and beta
        """
        length = len(pattern)
        remaining = set(self.alphabet) - guessed
        
        if not remaining:
            return {}
        
        if length not in self.hmms:
            return self._fallback_probabilities(remaining)
        
        hmm = self.hmms[length]
        B = hmm['B']
        
        # Run Forward-Backward
        alpha = self.forward(hmm, pattern, guessed)
        beta = self.backward(hmm, pattern, guessed)
        
        # Compute gamma: P(state_t | observations)
        gamma = alpha * beta
        gamma /= (gamma.sum(axis=1, keepdims=True) + 1e-10)
        
        # For each unknown position, compute letter probabilities
        letter_scores = defaultdict(float)
        unknown_positions = [i for i, c in enumerate(pattern) if c == '_']
        
        for pos in unknown_positions:
            # Use emission probabilities weighted by state probability
            state_prob = gamma[pos, pos]  # Probability of being at this state
            
            for letter in remaining:
                letter_idx = self.letter_to_idx[letter]
                # P(letter | position) * P(state | observations)
                letter_scores[letter] += B[pos, letter_idx] * state_prob
        
        # Normalize
        total = sum(letter_scores.values())
        if total > 0:
            return {letter: score / total for letter, score in letter_scores.items()}
        
        return self._fallback_probabilities(remaining)
    
    def _fallback_probabilities(self, remaining: Set[str]) -> Dict[str, float]:
        """Fallback when no HMM for this length."""
        if not self.hmms:
            # Use uniform
            return {letter: 1.0 / len(remaining) for letter in remaining}
        
        # Average emission probabilities across all HMMs
        letter_scores = defaultdict(float)
        
        for hmm in self.hmms.values():
            B = hmm['B']
            for letter in remaining:
                letter_idx = self.letter_to_idx[letter]
                # Average across all positions
                letter_scores[letter] += B[:, letter_idx].mean()
        
        total = sum(letter_scores.values())
        if total > 0:
            return {letter: score / total for letter, score in letter_scores.items()}
        
        return {letter: 1.0 / len(remaining) for letter in remaining}
    
    def choose_letter(self, pattern: str, guessed: Set[str]) -> str:
        """Choose letter with highest probability."""
        probs = self.get_letter_probabilities(pattern, guessed)
        
        if not probs:
            for letter in 'etaoinshrdlcumwfgypbvkjxqz':
                if letter not in guessed:
                    return letter
            return 'e'
        
        return max(probs.items(), key=lambda x: x[1])[0]


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


def play_game(hmm: ProbabilisticHMM, word: str) -> Tuple[bool, int, int]:
    """Play one game."""
    game = HangmanGame(word)
    
    while not game.is_done():
        letter = hmm.choose_letter(game.pattern, game.guessed)
        game.guess(letter)
    
    return (game.is_won(), game.wrong, game.repeated)


def evaluate(hmm: ProbabilisticHMM, test_file: str):
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
    print("RESULTS - VERSION 7 (Probabilistic HMM with Forward-Backward)")
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
    print("Hangman Agent V7 - Probabilistic HMM")
    print("Pure Forward-Backward Inference")
    print("=" * 70)
    
    corpus_file = 'data/corpus.txt'
    test_file = 'data/test.txt'
    
    # Train
    hmm = ProbabilisticHMM(max_length=24)
    hmm.train(corpus_file)
    
    # Evaluate
    score = evaluate(hmm, test_file)
    
    # Save
    with open('hmm_model_v7.pkl', 'wb') as f:
        pickle.dump(hmm, f)
    print("\nModel saved to hmm_model_v7.pkl")
    
    return score


if __name__ == "__main__":
    main()
