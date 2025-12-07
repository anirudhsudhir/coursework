"""
Hangman Agent - Optimized HMM
Version 9: Fine-tuned parameters and adaptive weighting

Improvements over V8:
- Better smoothing parameters
- Adaptive weighting based on available context
- Smarter position-aware probability combination
- Better handling of edge cases
"""

import numpy as np
from collections import defaultdict, Counter
from typing import List, Tuple, Set, Dict


class OptimizedHMM:
    """Optimized HMM with fine-tuned parameters."""
    
    def __init__(self, max_length: int = 24):
        self.max_length = max_length
        self.alphabet = 'abcdefghijklmnopqrstuvwxyz'
        self.n_letters = len(self.alphabet)
        self.letter_to_idx = {letter: i for i, letter in enumerate(self.alphabet)}
        
        # HMM emission matrices by length
        self.hmms = {}
        
        # N-gram models
        self.unigram_probs = {}
        self.bigram_probs = {}
        self.trigram_probs = {}
        
    def train(self, corpus_file: str):
        """Train all models."""
        print("Training Optimized HMM...")
        
        with open(corpus_file, 'r') as f:
            words = [line.strip().lower() for line in f if line.strip()]
        
        words_by_length = defaultdict(list)
        for word in words:
            if len(word) <= self.max_length:
                words_by_length[len(word)].append(word)
        
        print(f"Training on {len(words)} words")
        
        # Train HMMs
        for length in sorted(words_by_length.keys()):
            if len(words_by_length[length]) >= 5:
                self.hmms[length] = self._train_hmm(words_by_length[length], length)
        
        # Train n-grams
        self._train_ngrams(words)
        
        print(f"Trained HMMs for lengths: {sorted(self.hmms.keys())}")
    
    def _train_hmm(self, words: List[str], length: int) -> np.ndarray:
        """Train emission matrix for specific length."""
        emission_counts = np.zeros((length, self.n_letters))
        
        for word in words:
            for pos, letter in enumerate(word):
                if letter in self.letter_to_idx:
                    emission_counts[pos, self.letter_to_idx[letter]] += 1
        
        # Normalize with optimized smoothing
        B = np.zeros((length, self.n_letters))
        alpha = 0.3  # Reduced smoothing for more confidence
        for pos in range(length):
            smoothed = emission_counts[pos] + alpha
            B[pos] = smoothed / smoothed.sum()
        
        return B
    
    def _train_ngrams(self, words: List[str]):
        """Train n-gram models with better smoothing."""
        # Counts
        unigram_counts = Counter()
        bigram_counts = defaultdict(Counter)
        trigram_counts = defaultdict(Counter)
        
        for word in words:
            for letter in word:
                unigram_counts[letter] += 1
            
            for i in range(len(word) - 1):
                bigram_counts[word[i]][word[i+1]] += 1
            
            for i in range(len(word) - 2):
                bigram_counts[word[i:i+2]][word[i+2]] += 1
        
        # Unigram probabilities
        total = sum(unigram_counts.values())
        self.unigram_probs = {letter: count / total for letter, count in unigram_counts.items()}
        
        # Bigram probabilities with better smoothing
        for letter1 in self.alphabet:
            self.bigram_probs[letter1] = {}
            total_count = sum(bigram_counts[letter1].values())
            
            if total_count > 0:
                for letter2 in self.alphabet:
                    count = bigram_counts[letter1].get(letter2, 0)
                    # Use unigram prob as smoothing baseline
                    unigram_prob = self.unigram_probs.get(letter2, 1e-5)
                    smoothed = (count + 0.05 * unigram_prob * total_count) / (total_count + 0.05 * total_count)
                    self.bigram_probs[letter1][letter2] = smoothed
            else:
                # Fallback to unigrams
                for letter2 in self.alphabet:
                    self.bigram_probs[letter1][letter2] = self.unigram_probs.get(letter2, 1e-5)
        
        # Trigram probabilities
        for bigram, next_counts in trigram_counts.items():
            self.trigram_probs[bigram] = {}
            total_count = sum(next_counts.values())
            
            if total_count > 0:
                for letter3 in self.alphabet:
                    count = next_counts.get(letter3, 0)
                    unigram_prob = self.unigram_probs.get(letter3, 1e-5)
                    smoothed = (count + 0.05 * unigram_prob * total_count) / (total_count + 0.05 * total_count)
                    self.trigram_probs[bigram][letter3] = smoothed
    
    def get_letter_probabilities(self, pattern: str, guessed: Set[str]) -> Dict[str, float]:
        """
        Adaptively combine probability sources based on available context.
        """
        length = len(pattern)
        remaining = set(self.alphabet) - guessed
        
        if not remaining:
            return {}
        
        letter_scores = defaultdict(float)
        total_weight = 0.0
        
        # Count how much context we have
        revealed_count = sum(1 for c in pattern if c != '_')
        context_ratio = revealed_count / length if length > 0 else 0
        
        # Source 1: Position-based HMM (strongest for early game)
        if length in self.hmms:
            B = self.hmms[length]
            unknown_positions = [i for i, c in enumerate(pattern) if c == '_']
            
            position_weight = 4.0 * (1.0 + context_ratio)  # Increase weight as we get more context
            total_weight += position_weight
            
            for pos in unknown_positions:
                for letter in remaining:
                    letter_idx = self.letter_to_idx[letter]
                    letter_scores[letter] += B[pos, letter_idx] * position_weight
        
        # Source 2: Bigram context (strong when we have neighbors)
        bigram_weight_base = 3.0
        bigrams_found = 0
        
        for i, char in enumerate(pattern):
            if char != '_':
                # Right context
                if i + 1 < length and pattern[i + 1] == '_':
                    bigrams_found += 1
                    for letter in remaining:
                        prob = self.bigram_probs.get(char, {}).get(letter, 1e-5)
                        letter_scores[letter] += prob * bigram_weight_base
                
                # Left context
                if i - 1 >= 0 and pattern[i - 1] == '_':
                    bigrams_found += 1
                    for letter in remaining:
                        prob = self.bigram_probs.get(letter, {}).get(char, 1e-5)
                        letter_scores[letter] += prob * bigram_weight_base
        
        if bigrams_found > 0:
            total_weight += bigram_weight_base * bigrams_found
        
        # Source 3: Trigram context (very strong signal)
        trigram_weight = 4.0
        trigrams_found = 0
        
        for i in range(length - 2):
            # Pattern: A, B, ?
            if pattern[i] != '_' and pattern[i+1] != '_' and pattern[i+2] == '_':
                bigram = pattern[i:i+2]
                if bigram in self.trigram_probs:
                    trigrams_found += 1
                    for letter in remaining:
                        prob = self.trigram_probs[bigram].get(letter, 1e-5)
                        letter_scores[letter] += prob * trigram_weight
            
            # Pattern: A, ?, C
            if pattern[i] != '_' and pattern[i+1] == '_' and pattern[i+2] != '_':
                trigrams_found += 1
                for letter in remaining:
                    # P(letter|A) * P(C|letter)
                    prob1 = self.bigram_probs.get(pattern[i], {}).get(letter, 1e-5)
                    prob2 = self.bigram_probs.get(letter, {}).get(pattern[i+2], 1e-5)
                    letter_scores[letter] += (prob1 * prob2 * 100) * trigram_weight  # Scale up product
            
            # Pattern: ?, B, C
            if pattern[i] == '_' and pattern[i+1] != '_' and pattern[i+2] != '_':
                bigram = pattern[i+1:i+3]
                if bigram in self.trigram_probs:
                    # This is P(C | B), but we want P(?, B, C)
                    # Use P(? | reverse lookup)
                    for letter in remaining:
                        test_bigram = letter + pattern[i+1]
                        if test_bigram in self.trigram_probs:
                            prob = self.trigram_probs[test_bigram].get(pattern[i+2], 1e-5)
                            letter_scores[letter] += prob * trigram_weight
                    trigrams_found += 1
        
        if trigrams_found > 0:
            total_weight += trigram_weight * trigrams_found
        
        # Source 4: Unigram baseline (always present)
        unigram_weight = 1.0
        total_weight += unigram_weight
        
        for letter in remaining:
            prob = self.unigram_probs.get(letter, 1e-5)
            letter_scores[letter] += prob * unigram_weight
        
        # Normalize
        total_score = sum(letter_scores.values())
        if total_score > 0:
            return {letter: score / total_score for letter, score in letter_scores.items()}
        
        # Fallback
        return {letter: 1.0 / len(remaining) for letter in remaining}
    
    def choose_letter(self, pattern: str, guessed: Set[str]) -> str:
        """Choose best letter."""
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
                new_pattern.append(letter if c == letter else self.pattern[i])
            self.pattern = ''.join(new_pattern)
        else:
            self.wrong += 1
    
    def is_won(self):
        return '_' not in self.pattern
    
    def is_lost(self):
        return self.wrong >= self.max_wrong
    
    def is_done(self):
        return self.is_won() or self.is_lost()


def play_game(hmm: OptimizedHMM, word: str) -> Tuple[bool, int, int]:
    """Play one game."""
    game = HangmanGame(word)
    
    while not game.is_done():
        letter = hmm.choose_letter(game.pattern, game.guessed)
        game.guess(letter)
    
    return (game.is_won(), game.wrong, game.repeated)


def evaluate(hmm: OptimizedHMM, test_file: str):
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
    print("RESULTS - VERSION 9 (Optimized HMM)")
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
    
    if final_score > 0:
        print("\n🎉 POSITIVE SCORE ACHIEVED! 🎉")
    
    return final_score


def main():
    print("=" * 70)
    print("Hangman Agent V9 - Optimized HMM")
    print("Fine-tuned Parameters & Adaptive Weighting")
    print("=" * 70)
    
    corpus_file = 'data/corpus.txt'
    test_file = 'data/test.txt'
    
    # Train
    hmm = OptimizedHMM(max_length=24)
    hmm.train(corpus_file)
    
    # Evaluate
    score = evaluate(hmm, test_file)
    
    return score


if __name__ == "__main__":
    main()
