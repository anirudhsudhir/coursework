"""
Hangman Agent - Final Optimized Version
Version 10: Maximum Performance

Strategy:
1. Use HMM emission probabilities for position-based inference
2. Strong n-gram conditional probabilities (bigrams, trigrams)
3. Adaptive weighting that heavily favors trigram context
4. Optimized smoothing and probability combination
5. Smart early-game strategy focusing on high-value letters
"""

import numpy as np
from collections import defaultdict, Counter
from typing import List, Tuple, Set, Dict


class FinalHMM:
    """Final optimized HMM implementation."""
    
    def __init__(self):
        self.alphabet = 'abcdefghijklmnopqrstuvwxyz'
        self.letter_to_idx = {l: i for i, l in enumerate(self.alphabet)}
        
        # Models
        self.emission_probs = {}  # By length
        self.unigram = {}
        self.bigram = {}
        self.trigram = {}
        
    def train(self, corpus_file: str):
        """Train all models."""
        print("Training Final Optimized HMM...")
        
        with open(corpus_file, 'r') as f:
            words = [line.strip().lower() for line in f if line.strip()]
        
        # Group by length
        by_length = defaultdict(list)
        for w in words:
            by_length[len(w)].append(w)
        
        # Train emissions
        for length, word_list in by_length.items():
            if len(word_list) >= 3:
                self.emission_probs[length] = self._train_emissions(word_list, length)
        
        # Train n-grams
        self._train_ngrams(words)
        
        print(f"Trained on {len(words)} words, lengths: {sorted(self.emission_probs.keys())}")
    
    def _train_emissions(self, words: List[str], length: int) -> np.ndarray:
        """Train emission matrix."""
        B = np.zeros((length, 26))
        
        for word in words:
            for pos, letter in enumerate(word):
                if letter in self.letter_to_idx:
                    B[pos, self.letter_to_idx[letter]] += 1
        
        # Normalize with minimal smoothing for confidence
        for pos in range(length):
            B[pos] = (B[pos] + 0.1) / (B[pos].sum() + 2.6)
        
        return B
    
    def _train_ngrams(self, words: List[str]):
        """Train n-gram models."""
        uni_count = Counter()
        bi_count = defaultdict(Counter)
        tri_count = defaultdict(Counter)
        
        for word in words:
            for letter in word:
                uni_count[letter] += 1
            
            for i in range(len(word) - 1):
                bi_count[word[i]][word[i+1]] += 1
            
            for i in range(len(word) - 2):
                tri_count[word[i:i+2]][word[i+2]] += 1
        
        # Convert to probabilities
        total = sum(uni_count.values())
        for letter in self.alphabet:
            self.unigram[letter] = uni_count.get(letter, 0) / total
        
        # Bigrams with Good-Turing smoothing approximation
        for l1 in self.alphabet:
            self.bigram[l1] = {}
            total_count = sum(bi_count[l1].values())
            
            if total_count > 0:
                for l2 in self.alphabet:
                    count = bi_count[l1].get(l2, 0)
                    # Interpolation with unigram
                    lambda_param = 0.9  # Weight for bigram
                    self.bigram[l1][l2] = (
                        lambda_param * (count / total_count) +
                        (1 - lambda_param) * self.unigram.get(l2, 1e-5)
                    )
            else:
                for l2 in self.alphabet:
                    self.bigram[l1][l2] = self.unigram.get(l2, 1e-5)
        
        # Trigrams
        for bigram, next_counts in tri_count.items():
            self.trigram[bigram] = {}
            total_count = sum(next_counts.values())
            
            if total_count > 0:
                for l3 in self.alphabet:
                    count = next_counts.get(l3, 0)
                    # Interpolation with bigram
                    lambda_param = 0.85
                    second_letter = bigram[1] if len(bigram) > 1 else bigram[0]
                    fallback = self.bigram.get(second_letter, {}).get(l3, self.unigram.get(l3, 1e-5))
                    self.trigram[bigram][l3] = (
                        lambda_param * (count / total_count) +
                        (1 - lambda_param) * fallback
                    )
    
    def get_letter_probabilities(self, pattern: str, guessed: Set[str]) -> Dict[str, float]:
        """Get probabilities with optimized combination."""
        length = len(pattern)
        remaining = set(self.alphabet) - guessed
        
        if not remaining:
            return {}
        
        scores = {letter: 0.0 for letter in remaining}
        weights_sum = 0.0
        
        # Count revealed letters
        revealed = sum(1 for c in pattern if c != '_')
        
        # 1. Position-based emissions
        if length in self.emission_probs:
            B = self.emission_probs[length]
            unknown_pos = [i for i, c in enumerate(pattern) if c == '_']
            
            # Weight increases with context
            pos_weight = 5.0 * (1.0 + 0.5 * revealed / length)
            weights_sum += pos_weight
            
            for pos in unknown_pos:
                for letter in remaining:
                    scores[letter] += B[pos, self.letter_to_idx[letter]] * pos_weight
        
        # 2. Trigram context (VERY STRONG)
        tri_weight = 8.0  # Highest weight
        tri_count = 0
        
        for i in range(length - 2):
            # Case: AB?
            if pattern[i] != '_' and pattern[i+1] != '_' and pattern[i+2] == '_':
                bigram = pattern[i:i+2]
                if bigram in self.trigram:
                    tri_count += 1
                    for letter in remaining:
                        scores[letter] += self.trigram[bigram].get(letter, 0) * tri_weight
            
            # Case: A?C
            elif pattern[i] != '_' and pattern[i+1] == '_' and pattern[i+2] != '_':
                tri_count += 1
                for letter in remaining:
                    # P(B|A) × P(C|B)
                    p1 = self.bigram.get(pattern[i], {}).get(letter, 0)
                    p2 = self.bigram.get(letter, {}).get(pattern[i+2], 0)
                    scores[letter] += (p1 * p2 * 200) * tri_weight  # Scale up product
            
            # Case: ?BC
            elif pattern[i] == '_' and pattern[i+1] != '_' and pattern[i+2] != '_':
                tri_count += 1
                for letter in remaining:
                    bigram_test = letter + pattern[i+1]
                    if bigram_test in self.trigram:
                        prob = self.trigram[bigram_test].get(pattern[i+2], 0)
                        scores[letter] += prob * tri_weight
        
        if tri_count > 0:
            weights_sum += tri_weight * tri_count
        
        # 3. Bigram context
        bi_weight = 4.0
        bi_count = 0
        
        for i, char in enumerate(pattern):
            if char != '_':
                # Right neighbor
                if i + 1 < length and pattern[i + 1] == '_':
                    bi_count += 1
                    for letter in remaining:
                        scores[letter] += self.bigram.get(char, {}).get(letter, 0) * bi_weight
                
                # Left neighbor
                if i - 1 >= 0 and pattern[i - 1] == '_':
                    bi_count += 1
                    for letter in remaining:
                        scores[letter] += self.bigram.get(letter, {}).get(char, 0) * bi_weight
        
        if bi_count > 0:
            weights_sum += bi_weight * bi_count
        
        # 4. Unigram baseline
        uni_weight = 1.5
        weights_sum += uni_weight
        
        for letter in remaining:
            scores[letter] += self.unigram.get(letter, 1e-5) * uni_weight
        
        # Normalize
        total = sum(scores.values())
        if total > 0:
            return {letter: score / total for letter, score in scores.items()}
        
        return {letter: 1.0 / len(remaining) for letter in remaining}
    
    def choose_letter(self, pattern: str, guessed: Set[str]) -> str:
        """Choose best letter."""
        probs = self.get_letter_probabilities(pattern, guessed)
        
        if not probs:
            # Fallback
            for letter in 'etaoinshrdlcumwfgypbvkjxqz':
                if letter not in guessed:
                    return letter
            return 'e'
        
        return max(probs.items(), key=lambda x: x[1])[0]


class Hangman:
    """Hangman game."""
    
    def __init__(self, word: str):
        self.word = word.lower()
        self.pattern = '_' * len(word)
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
            self.pattern = ''.join(
                letter if self.word[i] == letter else self.pattern[i]
                for i in range(len(self.word))
            )
        else:
            self.wrong += 1
    
    def won(self):
        return '_' not in self.pattern
    
    def lost(self):
        return self.wrong >= 6


def play(hmm: FinalHMM, word: str) -> Tuple[bool, int, int]:
    """Play one game."""
    game = Hangman(word)
    
    while not game.won() and not game.lost():
        letter = hmm.choose_letter(game.pattern, game.guessed)
        game.guess(letter)
    
    return (game.won(), game.wrong, game.repeated)


def evaluate(hmm: FinalHMM, test_file: str):
    """Evaluate."""
    print("\nEvaluating...")
    
    with open(test_file, 'r') as f:
        words = [line.strip().lower() for line in f if line.strip()]
    
    wins, wrong, repeated = 0, 0, 0
    
    for i, word in enumerate(words):
        w, wr, r = play(hmm, word)
        wins += w
        wrong += wr
        repeated += r
        
        if (i + 1) % 500 == 0:
            print(f"  {i+1}/2000: {100*wins/(i+1):.2f}% success")
    
    n = len(words)
    rate = wins / n
    score = (rate * 2000) - (wrong * 5) - (repeated * 2)
    
    print("\n" + "=" * 70)
    print("FINAL RESULTS - VERSION 10")
    print("=" * 70)
    print(f"Success Rate: {rate*100:.2f}% ({wins}/{n})")
    print(f"Avg Wrong: {wrong/n:.2f}, Total Wrong: {wrong}")
    print(f"Avg Repeated: {repeated/n:.2f}, Total Repeated: {repeated}")
    print(f"\nFINAL SCORE: {score:.2f}")
    print("=" * 70)
    print(f"  Success:  +{rate * 2000:.2f}")
    print(f"  Wrong:    -{wrong * 5:.2f}")
    print(f"  Repeated: -{repeated * 2:.2f}")
    print("=" * 70)
    
    if score > 0:
        print("\n🎉 POSITIVE SCORE ACHIEVED! 🎉")
    else:
        print(f"\nNeed {abs(score)/2000*100:.1f}% more success rate for positive score")
    
    return score


def main():
    print("=" * 70)
    print("Hangman V10 - Final Optimized")
    print("=" * 70)
    
    hmm = FinalHMM()
    hmm.train('data/corpus.txt')
    score = evaluate(hmm, 'data/test.txt')
    
    return score


if __name__ == "__main__":
    main()
