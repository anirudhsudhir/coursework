"""
Hangman Agent - Enhanced Probabilistic HMM
Version 8: HMM + Contextual N-gram Probabilities

Combining:
1. HMM Forward-Backward for position-based probabilities
2. Conditional probabilities from bigrams/trigrams
3. Bayesian combination of multiple probability sources
"""

import numpy as np
from collections import defaultdict, Counter
from typing import List, Tuple, Set, Dict
import pickle


class EnhancedProbabilisticHMM:
    """
    Enhanced HMM with n-gram conditional probabilities.
    """
    
    def __init__(self, max_length: int = 24):
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


def play_game(hmm: EnhancedProbabilisticHMM, word: str) -> Tuple[bool, int, int]:
    """Play one game."""
    game = HangmanGame(word)
    
    while not game.is_done():
        letter = hmm.choose_letter(game.pattern, game.guessed)
        game.guess(letter)
    
    return (game.is_won(), game.wrong, game.repeated)


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
    print("Hangman Agent V8 - Enhanced Probabilistic HMM")
    print("HMM + N-gram Conditional Probabilities")
    print("=" * 70)
    
    corpus_file = 'data/corpus.txt'
    test_file = 'data/test.txt'
    
    # Train
    hmm = EnhancedProbabilisticHMM(max_length=24)
    hmm.train(corpus_file)
    
    # Evaluate
    score = evaluate(hmm, test_file)
    
    # Save
    with open('hmm_model_v8.pkl', 'wb') as f:
        pickle.dump(hmm, f)
    print("\nModel saved to hmm_model_v8.pkl")
    
    return score


if __name__ == "__main__":
    main()
