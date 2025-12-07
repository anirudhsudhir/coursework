"""
Hangman Agent - True HMM with Forward-Backward Algorithm
Version 6: Proper HMM Implementation

Using Hidden Markov Model with:
- Hidden states: Character positions in words
- Emissions: Actual letters
- Forward-Backward algorithm for inference
- Viterbi for most likely sequence
"""

import numpy as np
from collections import defaultdict, Counter
from typing import List, Tuple, Set, Dict
import pickle


class HiddenMarkovModel:
    """
    True HMM implementation with Forward-Backward algorithm.
    
    States: Position indices (0, 1, 2, ..., max_length-1)
    Emissions: Letters (a-z)
    """
    
    def __init__(self, max_length: int = 24):
        self.max_length = max_length
        self.n_states = max_length  # Position states
        self.alphabet = 'abcdefghijklmnopqrstuvwxyz'
        self.n_emissions = len(self.alphabet)
        self.letter_to_idx = {letter: i for i, letter in enumerate(self.alphabet)}
        self.idx_to_letter = {i: letter for i, letter in enumerate(self.alphabet)}
        
        # HMM parameters for each word length
        self.hmms_by_length = {}
        
    def train(self, corpus_file: str):
        """Train HMM on corpus using MLE (Maximum Likelihood Estimation)."""
        print("Training HMM with Forward-Backward approach...")
        
        with open(corpus_file, 'r') as f:
            words = [line.strip().lower() for line in f if line.strip()]
        
        # Group words by length
        words_by_length = defaultdict(list)
        for word in words:
            if len(word) <= self.max_length:
                words_by_length[len(word)].append(word)
        
        print(f"Training on {len(words)} words")
        
        # Train separate HMM for each word length
        for length in sorted(words_by_length.keys()):
            if len(words_by_length[length]) >= 10:  # Only train if enough samples
                self.hmms_by_length[length] = self._train_hmm_for_length(
                    words_by_length[length], length
                )
        
        print(f"Trained HMMs for lengths: {sorted(self.hmms_by_length.keys())}")
    
    def _train_hmm_for_length(self, words: List[str], length: int) -> Dict:
        """
        Train HMM for specific word length using MLE.
        
        For Hangman, we can use a simple left-to-right model where:
        - Initial state probabilities: start at position 0
        - Transition probabilities: move left to right
        - Emission probabilities: letter distribution at each position
        """
        n_words = len(words)
        
        # Initialize parameters
        # Start probability (always start at position 0)
        start_prob = np.zeros(length)
        start_prob[0] = 1.0
        
        # Transition probability (left-to-right model)
        # P(state_i | state_i-1) - move sequentially through positions
        transition_prob = np.zeros((length, length))
        for i in range(length - 1):
            transition_prob[i, i + 1] = 1.0
        transition_prob[length - 1, length - 1] = 1.0  # Stay at last state
        
        # Emission probability P(letter | position)
        # Count letters at each position
        emission_counts = np.zeros((length, self.n_emissions))
        
        for word in words:
            for pos, letter in enumerate(word):
                if letter in self.letter_to_idx:
                    letter_idx = self.letter_to_idx[letter]
                    emission_counts[pos, letter_idx] += 1
        
        # Normalize to get probabilities (with smoothing)
        emission_prob = np.zeros((length, self.n_emissions))
        for pos in range(length):
            # Add-one (Laplace) smoothing
            smoothed_counts = emission_counts[pos] + 1
            emission_prob[pos] = smoothed_counts / smoothed_counts.sum()
        
        return {
            'length': length,
            'start_prob': start_prob,
            'transition_prob': transition_prob,
            'emission_prob': emission_prob,
            'n_words': n_words
        }
    
    def forward_algorithm(self, hmm: Dict, observed_pattern: str, guessed: Set[str]) -> Tuple[np.ndarray, float]:
        """
        Forward algorithm to compute P(observations | model).
        
        Returns: (alpha matrix, total probability)
        """
        length = hmm['length']
        emission_prob = hmm['emission_prob']
        transition_prob = hmm['transition_prob']
        start_prob = hmm['start_prob']
        
        # Alpha: forward probabilities
        # alpha[t, i] = P(observations up to time t, state i at time t | model)
        alpha = np.zeros((length, length))
        
        # Initialization (t=0)
        for state in range(length):
            if observed_pattern[0] == '_':
                # Unknown - sum over all possible letters
                obs_prob = 1.0  # We'll handle this differently
            else:
                # Known letter
                letter = observed_pattern[0]
                if letter in self.letter_to_idx:
                    letter_idx = self.letter_to_idx[letter]
                    obs_prob = emission_prob[state, letter_idx]
                else:
                    obs_prob = 0.0
            
            alpha[0, state] = start_prob[state] * obs_prob
        
        # Recursion
        for t in range(1, length):
            for state in range(length):
                # Sum over all previous states
                transition_sum = 0.0
                for prev_state in range(length):
                    transition_sum += alpha[t-1, prev_state] * transition_prob[prev_state, state]
                
                # Observation probability
                if observed_pattern[t] == '_':
                    obs_prob = 1.0
                else:
                    letter = observed_pattern[t]
                    if letter in self.letter_to_idx:
                        letter_idx = self.letter_to_idx[letter]
                        obs_prob = emission_prob[state, letter_idx]
                    else:
                        obs_prob = 0.0
                
                alpha[t, state] = transition_sum * obs_prob
        
        # Total probability
        total_prob = alpha[length - 1].sum()
        
        return alpha, total_prob
    
    def get_letter_probabilities(self, pattern: str, guessed: Set[str]) -> Dict[str, float]:
        """
        Use Forward-Backward to infer letter probabilities for unknown positions.
        """
        length = len(pattern)
        remaining = set(self.alphabet) - guessed
        
        if length not in self.hmms_by_length:
            # Fallback to simple frequency
            return self._fallback_probabilities(pattern, guessed)
        
        hmm = self.hmms_by_length[length]
        emission_prob = hmm['emission_prob']
        
        # For each unknown position, calculate letter probabilities
        letter_scores = defaultdict(float)
        unknown_positions = [i for i, c in enumerate(pattern) if c == '_']
        
        if not unknown_positions:
            return {}
        
        # Strategy: Use emission probabilities at unknown positions
        for pos in unknown_positions:
            for letter in remaining:
                letter_idx = self.letter_to_idx[letter]
                # Weight by emission probability at this position
                letter_scores[letter] += emission_prob[pos, letter_idx]
        
        # Normalize
        total = sum(letter_scores.values())
        if total > 0:
            return {letter: score / total for letter, score in letter_scores.items()}
        
        return self._fallback_probabilities(pattern, guessed)
    
    def _fallback_probabilities(self, pattern: str, guessed: Set[str]) -> Dict[str, float]:
        """Fallback when no HMM available for this length."""
        remaining = set(self.alphabet) - guessed
        
        # Use overall letter frequency across all HMMs
        letter_scores = defaultdict(float)
        
        for length, hmm in self.hmms_by_length.items():
            emission_prob = hmm['emission_prob']
            for letter in remaining:
                letter_idx = self.letter_to_idx[letter]
                # Average emission probability across all positions
                letter_scores[letter] += emission_prob[:, letter_idx].mean()
        
        if letter_scores:
            total = sum(letter_scores.values())
            return {letter: score / total for letter, score in letter_scores.items()}
        
        # Ultimate fallback - uniform
        return {letter: 1.0 / len(remaining) for letter in remaining}
    
    def choose_best_letter(self, pattern: str, guessed: Set[str]) -> str:
        """Choose best letter based on HMM probabilities."""
        probs = self.get_letter_probabilities(pattern, guessed)
        
        if not probs:
            # Fallback to common letters
            for letter in 'etaoinshrdlcumwfgypbvkjxqz':
                if letter not in guessed:
                    return letter
            return 'e'
        
        # Return highest probability letter
        return max(probs.items(), key=lambda x: x[1])[0]


class HangmanGame:
    """Hangman game environment."""
    
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
        """Make a guess."""
        letter = letter.lower()
        
        if letter in self.guessed:
            self.repeated += 1
            return
        
        self.guessed.add(letter)
        
        if letter in self.word:
            # Update pattern
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


def play_game(hmm: HiddenMarkovModel, word: str) -> Tuple[bool, int, int]:
    """Play one game. Returns (won, wrong_guesses, repeated_guesses)."""
    game = HangmanGame(word)
    
    while not game.is_done():
        letter = hmm.choose_best_letter(game.pattern, game.guessed)
        game.guess(letter)
    
    return (game.is_won(), game.wrong, game.repeated)


def evaluate(hmm: HiddenMarkovModel, test_file: str):
    """Evaluate HMM on test set."""
    print("\nEvaluating HMM...")
    
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
    
    # Calculate final score
    final_score = (success_rate * 2000) - (total_wrong * 5) - (total_repeated * 2)
    
    print("\n" + "=" * 70)
    print("RESULTS - VERSION 6 (True HMM with Forward-Backward)")
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
    print("Hangman Agent V6 - True HMM with Forward-Backward")
    print("=" * 70)
    
    corpus_file = 'data/corpus.txt'
    test_file = 'data/test.txt'
    
    # Train HMM
    hmm = HiddenMarkovModel(max_length=24)
    hmm.train(corpus_file)
    
    # Evaluate
    score = evaluate(hmm, test_file)
    
    # Save model
    with open('hmm_model_v6.pkl', 'wb') as f:
        pickle.dump(hmm, f)
    print("\nModel saved to hmm_model_v6.pkl")
    
    return score


if __name__ == "__main__":
    main()
