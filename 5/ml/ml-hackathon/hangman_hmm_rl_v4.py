"""
Hangman Agent - Pattern Matching Focus
Version 4: Pure Pattern Matching
- Focus on exact pattern matching
- Use frequency from matching words only
- No forced strategies, let data guide decisions
"""

import re
from collections import Counter
from typing import Set, Dict, Tuple


class PatternMatcher:
    """Pure pattern-based word matching."""
    
    def __init__(self):
        self.words_by_length = {}
        self.all_words = []
        
    def train(self, corpus_file: str):
        """Load corpus."""
        print("Loading corpus...")
        
        with open(corpus_file, 'r') as f:
            self.all_words = [line.strip().lower() for line in f if line.strip()]
        
        # Index by length
        for word in self.all_words:
            length = len(word)
            if length not in self.words_by_length:
                self.words_by_length[length] = []
            self.words_by_length[length].append(word)
        
        print(f"Loaded {len(self.all_words)} words")
        print(f"Length range: {min(self.words_by_length.keys())}-{max(self.words_by_length.keys())}")
    
    def find_matches(self, pattern: str, guessed: Set[str]) -> list:
        """Find all words matching pattern."""
        length = len(pattern)
        candidates = self.words_by_length.get(length, [])
        
        matches = []
        for word in candidates:
            if self._matches(word, pattern, guessed):
                matches.append(word)
        
        return matches
    
    def _matches(self, word: str, pattern: str, guessed: Set[str]) -> bool:
        """Check if word matches pattern and constraints."""
        for w_char, p_char in zip(word, pattern):
            if p_char != '_':
                if w_char != p_char:
                    return False
            else:
                if w_char in guessed:
                    return False
        return True
    
    def get_best_letter(self, pattern: str, guessed: Set[str]) -> str:
        """Get best letter to guess based on matching words."""
        remaining = set('abcdefghijklmnopqrstuvwxyz') - guessed
        
        if not remaining:
            return 'e'
        
        # Find matching words
        matches = self.find_matches(pattern, guessed)
        
        if not matches:
            # No matches - use common letters
            for letter in 'etaoinshrdlcumwfgypbvkjxqz':
                if letter in remaining:
                    return letter
        
        # Count letter frequencies in unknown positions
        letter_counts = Counter()
        unknown_pos = [i for i, c in enumerate(pattern) if c == '_']
        
        for word in matches:
            seen_in_word = set()
            for pos in unknown_pos:
                letter = word[pos]
                if letter in remaining and letter not in seen_in_word:
                    letter_counts[letter] += 1
                    seen_in_word.add(letter)
        
        if letter_counts:
            return letter_counts.most_common(1)[0][0]
        
        # Fallback
        return list(remaining)[0]


class SimpleHangman:
    """Hangman game."""
    
    def __init__(self, word: str):
        self.word = word.lower()
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
            self.pattern = ''.join(
                c if c != '_' else (letter if self.word[i] == letter else '_')
                for i, c in enumerate(self.pattern)
            )
        else:
            self.wrong += 1
    
    def won(self):
        return '_' not in self.pattern
    
    def lost(self):
        return self.wrong >= 6


def play_game(matcher: PatternMatcher, word: str) -> Tuple[bool, int, int]:
    """Play one game."""
    game = SimpleHangman(word)
    
    while not game.won() and not game.lost():
        letter = matcher.get_best_letter(game.pattern, game.guessed)
        game.guess(letter)
    
    return (game.won(), game.wrong, game.repeated)


def evaluate(matcher: PatternMatcher, test_file: str):
    """Evaluate on test set."""
    print("\nEvaluating...")
    
    with open(test_file, 'r') as f:
        words = [line.strip().lower() for line in f if line.strip()]
    
    wins = 0
    total_wrong = 0
    total_repeated = 0
    
    for i, word in enumerate(words):
        won, wrong, repeated = play_game(matcher, word)
        wins += won
        total_wrong += wrong
        total_repeated += repeated
        
        if (i + 1) % 500 == 0:
            print(f"  {i+1}/{len(words)}: {100*wins/(i+1):.2f}% success")
    
    n = len(words)
    success_rate = wins / n
    score = (success_rate * 2000) - (total_wrong * 5) - (total_repeated * 2)
    
    print("\n" + "=" * 70)
    print("RESULTS - VERSION 4")
    print("=" * 70)
    print(f"Success Rate: {100*success_rate:.2f}% ({wins}/{n})")
    print(f"Avg Wrong: {total_wrong/n:.2f}")
    print(f"Avg Repeated: {total_repeated/n:.2f}")
    print(f"Total Wrong: {total_wrong}")
    print(f"Total Repeated: {total_repeated}")
    print(f"\nFINAL SCORE: {score:.2f}")
    print("=" * 70)
    
    return score


def main():
    print("=" * 70)
    print("Version 4 - Pure Pattern Matching")
    print("=" * 70)
    
    matcher = PatternMatcher()
    matcher.train('data/corpus.txt')
    score = evaluate(matcher, 'data/test.txt')
    
    return score


if __name__ == "__main__":
    main()
