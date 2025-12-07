"""
Hangman Agent using Advanced Pattern Matching
Version 3: Sophisticated Strategy
- Vowel-first strategy
- N-gram pattern analysis
- Better probability weighting
- Adaptive strategy based on word length and progress
"""

import random
from collections import defaultdict, Counter
from typing import List, Tuple, Set, Dict
import re


class AdvancedHMM:
    """
    Advanced pattern matching with n-gram analysis.
    """
    
    def __init__(self):
        self.words_by_length = defaultdict(set)
        self.all_words = set()
        self.letter_freq = Counter()
        self.bigram_freq = defaultdict(Counter)
        self.trigram_freq = defaultdict(Counter)
        
    def train(self, corpus_file: str):
        """Train on corpus."""
        print("Training Advanced HMM...")
        
        with open(corpus_file, 'r') as f:
            words = [line.strip().lower() for line in f if line.strip()]
        
        for word in words:
            self.all_words.add(word)
            self.words_by_length[len(word)].add(word)
            
            # Letter frequencies
            for letter in word:
                self.letter_freq[letter] += 1
            
            # Bigram frequencies
            for i in range(len(word) - 1):
                bigram = word[i:i+2]
                self.bigram_freq[word[i]][word[i+1]] += 1
            
            # Trigram frequencies
            for i in range(len(word) - 2):
                trigram = word[i:i+3]
                self.trigram_freq[trigram[:2]][word[i+2]] += 1
        
        print(f"Trained on {len(words)} words")
        print(f"Unique words: {len(self.all_words)}")
        
    def get_matching_words(self, pattern: str, guessed_letters: Set[str]) -> Set[str]:
        """Get all words matching the pattern."""
        length = len(pattern)
        candidates = self.words_by_length.get(length, set())
        
        # Build regex pattern
        regex_pattern = ''
        for char in pattern:
            if char == '_':
                # Must not be a guessed letter
                excluded = ''.join(sorted(guessed_letters))
                if excluded:
                    regex_pattern += f'[^{excluded}]'
                else:
                    regex_pattern += '.'
            else:
                regex_pattern += char
        
        regex = re.compile(f'^{regex_pattern}$')
        
        matching = {word for word in candidates if regex.match(word)}
        return matching
    
    def get_letter_probabilities(self, pattern: str, guessed_letters: Set[str]) -> Dict[str, float]:
        """Get probability distribution over letters."""
        remaining = set('abcdefghijklmnopqrstuvwxyz') - guessed_letters
        
        if not remaining:
            return {}
        
        # Find matching words
        matching_words = self.get_matching_words(pattern, guessed_letters)
        
        if matching_words:
            # Count letters in unknown positions
            letter_counts = Counter()
            unknown_positions = [i for i, c in enumerate(pattern) if c == '_']
            
            for word in matching_words:
                for pos in unknown_positions:
                    if pos < len(word):
                        letter = word[pos]
                        if letter in remaining:
                            letter_counts[letter] += 1
            
            # Also consider overall letter frequency in matching words
            for word in matching_words:
                for letter in word:
                    if letter in remaining:
                        letter_counts[letter] += 0.5  # Lower weight for overall freq
            
            if letter_counts:
                total = sum(letter_counts.values())
                return {letter: count / total for letter, count in letter_counts.items()}
        
        # Fallback: use global letter frequencies
        letter_scores = {letter: self.letter_freq.get(letter, 1) for letter in remaining}
        total = sum(letter_scores.values())
        return {letter: score / total for letter, score in letter_scores.items()}


class AdaptiveAgent:
    """
    Adaptive agent with multi-stage strategy.
    """
    
    def __init__(self, hmm: AdvancedHMM):
        self.hmm = hmm
        # Common letter order based on English frequency
        self.vowels = 'aeiou'
        self.common_consonants = 'tnshrdlcmwfgypbvkjxqz'
        
    def choose_action(self, state: Tuple[str, Set[str], int]) -> str:
        """Choose best letter with adaptive strategy."""
        pattern, guessed_letters, wrong_count = state
        remaining = set('abcdefghijklmnopqrstuvwxyz') - guessed_letters
        
        if not remaining:
            return 'e'
        
        # Stage 1: Start with vowels if few guesses made
        if len(guessed_letters) < 3:
            for vowel in self.vowels:
                if vowel in remaining:
                    return vowel
        
        # Stage 2: Use HMM for pattern-based selection
        probs = self.hmm.get_letter_probabilities(pattern, guessed_letters)
        
        if probs:
            # Get top candidates
            sorted_letters = sorted(probs.items(), key=lambda x: x[1], reverse=True)
            
            # If we have high-confidence letter, use it
            if sorted_letters[0][1] > 0.3:  # High confidence threshold
                return sorted_letters[0][0]
            
            # Otherwise, consider top 3 and prefer common letters
            top_letters = [l for l, _ in sorted_letters[:5]]
            
            # Prefer common letters among top choices
            for letter in self.vowels + self.common_consonants:
                if letter in top_letters:
                    return letter
            
            # Return highest probability
            return sorted_letters[0][0]
        
        # Stage 3: Fallback to common letters
        for letter in self.vowels + self.common_consonants:
            if letter in remaining:
                return letter
        
        # Ultimate fallback
        return random.choice(list(remaining))


class HangmanGame:
    """Simple Hangman game."""
    
    def __init__(self, word: str, max_wrong: int = 6):
        self.word = word.lower()
        self.max_wrong = max_wrong
        self.reset()
    
    def reset(self):
        self.guessed = set()
        self.wrong = 0
        self.repeated = 0
        self.pattern = '_' * len(self.word)
    
    def guess(self, letter: str) -> bool:
        """Make a guess. Returns True if correct."""
        letter = letter.lower()
        
        if letter in self.guessed:
            self.repeated += 1
            return False
        
        self.guessed.add(letter)
        
        if letter in self.word:
            # Update pattern
            new_pattern = list(self.pattern)
            for i, char in enumerate(self.word):
                if char == letter:
                    new_pattern[i] = letter
            self.pattern = ''.join(new_pattern)
            return True
        else:
            self.wrong += 1
            return False
    
    def is_won(self) -> bool:
        return '_' not in self.pattern
    
    def is_lost(self) -> bool:
        return self.wrong >= self.max_wrong
    
    def is_done(self) -> bool:
        return self.is_won() or self.is_lost()
    
    def get_state(self):
        return (self.pattern, self.guessed.copy(), self.wrong)


def play_game(agent: AdaptiveAgent, word: str) -> Tuple[bool, int, int]:
    """
    Play one game.
    Returns: (won, wrong_guesses, repeated_guesses)
    """
    game = HangmanGame(word)
    
    while not game.is_done():
        state = game.get_state()
        action = agent.choose_action(state)
        game.guess(action)
    
    return (game.is_won(), game.wrong, game.repeated)


def evaluate_agent(agent: AdaptiveAgent, test_file: str) -> Dict:
    """Evaluate agent on test set."""
    print("\nEvaluating agent...")
    
    with open(test_file, 'r') as f:
        test_words = [line.strip().lower() for line in f if line.strip()]
    
    wins = 0
    total_wrong = 0
    total_repeated = 0
    
    for i, word in enumerate(test_words):
        won, wrong, repeated = play_game(agent, word)
        
        if won:
            wins += 1
        total_wrong += wrong
        total_repeated += repeated
        
        if (i + 1) % 500 == 0:
            success_rate = wins / (i + 1)
            print(f"  Progress: {i+1}/{len(test_words)} games, Success Rate: {success_rate*100:.2f}%")
    
    total_games = len(test_words)
    success_rate = wins / total_games
    avg_wrong = total_wrong / total_games
    avg_repeated = total_repeated / total_games
    
    # Calculate final score
    final_score = (success_rate * 2000) - (total_wrong * 5) - (total_repeated * 2)
    
    return {
        'success_rate': success_rate,
        'wins': wins,
        'total_games': total_games,
        'avg_wrong': avg_wrong,
        'avg_repeated': avg_repeated,
        'total_wrong': total_wrong,
        'total_repeated': total_repeated,
        'final_score': final_score
    }


def main():
    """Main execution."""
    print("=" * 70)
    print("Hangman Agent - Advanced Strategy (Version 3)")
    print("=" * 70)
    
    corpus_file = 'data/corpus.txt'
    test_file = 'data/test.txt'
    
    # Train HMM
    hmm = AdvancedHMM()
    hmm.train(corpus_file)
    
    # Create agent
    agent = AdaptiveAgent(hmm)
    
    # Evaluate
    results = evaluate_agent(agent, test_file)
    
    # Print results
    print("\n" + "=" * 70)
    print("EVALUATION RESULTS - VERSION 3")
    print("=" * 70)
    print(f"Success Rate: {results['success_rate']*100:.2f}% ({results['wins']}/{results['total_games']})")
    print(f"Average Wrong Guesses: {results['avg_wrong']:.2f}")
    print(f"Average Repeated Guesses: {results['avg_repeated']:.2f}")
    print(f"Total Wrong Guesses: {results['total_wrong']}")
    print(f"Total Repeated Guesses: {results['total_repeated']}")
    print(f"\nFINAL SCORE: {results['final_score']:.2f}")
    print("=" * 70)
    
    # Score breakdown
    print("\nScore Breakdown:")
    print(f"  Success bonus: {results['success_rate'] * 2000:.2f}")
    print(f"  Wrong penalty: -{results['total_wrong'] * 5:.2f}")
    print(f"  Repeated penalty: -{results['total_repeated'] * 2:.2f}")
    
    return results


if __name__ == "__main__":
    main()
