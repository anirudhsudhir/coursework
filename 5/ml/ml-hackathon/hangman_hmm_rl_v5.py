"""
Hangman Agent - Statistical Pattern Learning
Version 5: Generalized Learning (No Word Memorization)

Key insight: Test set has ZERO overlap with corpus
Strategy: Learn statistical patterns, not specific words
- Position-based letter frequencies
- N-gram patterns (bigrams, trigrams)
- Conditional probabilities
- Pattern generalization
"""

from collections import defaultdict, Counter
from typing import Set, Tuple, Dict
import math


class StatisticalPatternLearner:
    """
    Learn statistical patterns from corpus without memorizing words.
    """
    
    def __init__(self):
        # Position-based statistics
        self.position_freq = defaultdict(lambda: defaultdict(lambda: defaultdict(int)))  # [length][position][letter] = count
        self.length_letter_freq = defaultdict(Counter)  # [length][letter] = count
        
        # N-gram patterns
        self.bigram_freq = defaultdict(Counter)  # [letter1][letter2] = count
        self.trigram_freq = defaultdict(lambda: Counter())  # [letter1+letter2][letter3] = count
        
        # Global statistics
        self.global_letter_freq = Counter()
        self.word_count_by_length = Counter()
        
    def train(self, corpus_file: str):
        """Learn statistical patterns from corpus."""
        print("Learning statistical patterns from corpus...")
        
        with open(corpus_file, 'r') as f:
            words = [line.strip().lower() for line in f if line.strip()]
        
        for word in words:
            length = len(word)
            self.word_count_by_length[length] += 1
            
            # Position-based frequencies
            for pos, letter in enumerate(word):
                self.position_freq[length][pos][letter] += 1
                self.length_letter_freq[length][letter] += 1
                self.global_letter_freq[letter] += 1
            
            # Bigrams
            for i in range(len(word) - 1):
                self.bigram_freq[word[i]][word[i+1]] += 1
            
            # Trigrams
            for i in range(len(word) - 2):
                bigram = word[i:i+2]
                self.trigram_freq[bigram][word[i+2]] += 1
        
        print(f"Learned from {len(words)} words")
        print(f"Length distribution: {min(self.word_count_by_length.keys())} to {max(self.word_count_by_length.keys())}")
        print(f"Most common letters: {self.global_letter_freq.most_common(10)}")
    
    def get_letter_scores(self, pattern: str, guessed: Set[str]) -> Dict[str, float]:
        """
        Calculate scores for each remaining letter based on statistical patterns.
        """
        length = len(pattern)
        remaining = set('abcdefghijklmnopqrstuvwxyz') - guessed
        
        if not remaining:
            return {}
        
        letter_scores = defaultdict(float)
        
        # Strategy 1: Position-based scoring
        unknown_positions = [i for i, c in enumerate(pattern) if c == '_']
        for pos in unknown_positions:
            if length in self.position_freq and pos in self.position_freq[length]:
                pos_freq = self.position_freq[length][pos]
                total = sum(pos_freq.values())
                
                for letter in remaining:
                    if letter in pos_freq:
                        # Normalized frequency for this position
                        letter_scores[letter] += (pos_freq[letter] / total) * 10
        
        # Strategy 2: Context-based scoring (bigrams/trigrams)
        for i, char in enumerate(pattern):
            if char != '_':
                # Look at neighbors
                # Right neighbor
                if i + 1 < length and pattern[i + 1] == '_':
                    if char in self.bigram_freq:
                        total = sum(self.bigram_freq[char].values())
                        for letter in remaining:
                            if letter in self.bigram_freq[char]:
                                letter_scores[letter] += (self.bigram_freq[char][letter] / total) * 5
                
                # Left neighbor
                if i - 1 >= 0 and pattern[i - 1] == '_':
                    # Find letters that come before char
                    for prev_letter in remaining:
                        if prev_letter in self.bigram_freq and char in self.bigram_freq[prev_letter]:
                            total = sum(self.bigram_freq[prev_letter].values())
                            letter_scores[prev_letter] += (self.bigram_freq[prev_letter][char] / total) * 5
                
                # Trigram context
                if i + 2 < length and pattern[i + 1] != '_' and pattern[i + 2] == '_':
                    bigram = char + pattern[i + 1]
                    if bigram in self.trigram_freq:
                        total = sum(self.trigram_freq[bigram].values())
                        for letter in remaining:
                            if letter in self.trigram_freq[bigram]:
                                letter_scores[letter] += (self.trigram_freq[bigram][letter] / total) * 3
        
        # Strategy 3: Length-specific letter frequency
        if length in self.length_letter_freq:
            total = sum(self.length_letter_freq[length].values())
            for letter in remaining:
                if letter in self.length_letter_freq[length]:
                    letter_scores[letter] += (self.length_letter_freq[length][letter] / total) * 2
        
        # Strategy 4: Global letter frequency (fallback/baseline)
        total_global = sum(self.global_letter_freq.values())
        for letter in remaining:
            letter_scores[letter] += (self.global_letter_freq.get(letter, 1) / total_global) * 1
        
        # Normalize scores
        if letter_scores:
            max_score = max(letter_scores.values())
            if max_score > 0:
                return {letter: score / max_score for letter, score in letter_scores.items()}
        
        # Ultimate fallback
        return {letter: 1.0 / len(remaining) for letter in remaining}
    
    def choose_letter(self, pattern: str, guessed: Set[str]) -> str:
        """Choose best letter based on statistical patterns."""
        scores = self.get_letter_scores(pattern, guessed)
        
        if not scores:
            # Fallback to most common unguessed letter
            for letter in 'etaoinshrdlcumwfgypbvkjxqz':
                if letter not in guessed:
                    return letter
            return 'e'
        
        # Return letter with highest score
        return max(scores.items(), key=lambda x: x[1])[0]


class HangmanGame:
    """Simple Hangman game."""
    
    def __init__(self, word: str, max_wrong: int = 6):
        self.word = word.lower()
        self.max_wrong = max_wrong
        self.pattern = '_' * len(word)
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
            for i, c in enumerate(self.pattern):
                if c != '_':
                    new_pattern.append(c)
                elif self.word[i] == letter:
                    new_pattern.append(letter)
                else:
                    new_pattern.append('_')
            self.pattern = ''.join(new_pattern)
        else:
            self.wrong += 1
    
    def is_won(self):
        return '_' not in self.pattern
    
    def is_lost(self):
        return self.wrong >= self.max_wrong
    
    def is_done(self):
        return self.is_won() or self.is_lost()


def play_game(learner: StatisticalPatternLearner, word: str) -> Tuple[bool, int, int]:
    """Play one game and return (won, wrong_guesses, repeated_guesses)."""
    game = HangmanGame(word)
    
    while not game.is_done():
        letter = learner.choose_letter(game.pattern, game.guessed)
        game.guess(letter)
    
    return (game.is_won(), game.wrong, game.repeated)


def evaluate(learner: StatisticalPatternLearner, test_file: str):
    """Evaluate on test set."""
    print("\nEvaluating on test set...")
    
    with open(test_file, 'r') as f:
        test_words = [line.strip().lower() for line in f if line.strip()]
    
    wins = 0
    total_wrong = 0
    total_repeated = 0
    
    for i, word in enumerate(test_words):
        won, wrong, repeated = play_game(learner, word)
        
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
    print("RESULTS - VERSION 5 (Statistical Pattern Learning)")
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
    print("Hangman Agent V5 - Statistical Pattern Learning")
    print("No word memorization - Pure pattern generalization")
    print("=" * 70)
    
    corpus_file = 'data/corpus.txt'
    test_file = 'data/test.txt'
    
    # Train learner
    learner = StatisticalPatternLearner()
    learner.train(corpus_file)
    
    # Evaluate
    score = evaluate(learner, test_file)
    
    return score


if __name__ == "__main__":
    main()
