"""
Hangman Agent using Hidden Markov Model + Reinforcement Learning
Version 2: Improved Strategy
- Better HMM with position-aware probabilities
- Smarter letter selection combining frequency and position
- Improved reward structure
- Better exploration strategy
"""

import numpy as np
import random
from collections import defaultdict, Counter
import matplotlib.pyplot as plt
from typing import List, Tuple, Set, Dict


class ImprovedHMM:
    """
    Improved HMM with position-aware letter probabilities.
    """
    
    def __init__(self):
        self.words_by_length = defaultdict(list)
        self.letter_position_freq = defaultdict(lambda: defaultdict(lambda: defaultdict(int)))
        self.global_letter_freq = Counter()
        
    def train(self, corpus_file: str):
        """Train HMM on corpus of words."""
        print("Training Improved HMM on corpus...")
        
        with open(corpus_file, 'r') as f:
            words = [line.strip().lower() for line in f if line.strip()]
        
        # Group words by length and analyze patterns
        for word in words:
            length = len(word)
            self.words_by_length[length].append(word)
            
            # Track letter frequencies at each position
            for pos, letter in enumerate(word):
                self.letter_position_freq[length][pos][letter] += 1
                self.global_letter_freq[letter] += 1
        
        print(f"Trained on {len(words)} words")
        print(f"Word lengths: {min(self.words_by_length.keys())} to {max(self.words_by_length.keys())}")
        
    def get_letter_probabilities(self, masked_word: str, guessed_letters: Set[str]) -> Dict[str, float]:
        """
        Get sophisticated probability distribution over remaining letters.
        """
        length = len(masked_word)
        remaining_letters = set('abcdefghijklmnopqrstuvwxyz') - guessed_letters
        
        if not remaining_letters:
            return {}
        
        # Find all matching words
        matching_words = self._find_matching_words(masked_word, guessed_letters)
        
        if matching_words:
            # Strategy 1: Count letters in matching words (with position awareness)
            letter_scores = defaultdict(float)
            
            for word in matching_words:
                for pos, letter in enumerate(word):
                    if letter in remaining_letters and masked_word[pos] == '_':
                        # Score higher if letter appears in unknown position
                        letter_scores[letter] += 1.0
            
            # Normalize
            total = sum(letter_scores.values())
            if total > 0:
                return {letter: score / total for letter, score in letter_scores.items()}
        
        # Fallback: Use position-specific frequencies from training
        letter_scores = defaultdict(float)
        unknown_positions = [i for i, c in enumerate(masked_word) if c == '_']
        
        for pos in unknown_positions:
            if pos < length and length in self.letter_position_freq:
                for letter in remaining_letters:
                    freq = self.letter_position_freq[length][pos].get(letter, 0)
                    letter_scores[letter] += freq
        
        # If still no matches, use global frequencies
        if sum(letter_scores.values()) == 0:
            for letter in remaining_letters:
                letter_scores[letter] = self.global_letter_freq.get(letter, 1)
        
        # Normalize
        total = sum(letter_scores.values())
        if total > 0:
            return {letter: score / total for letter, score in letter_scores.items()}
        
        # Ultimate fallback: uniform
        return {letter: 1.0 / len(remaining_letters) for letter in remaining_letters}
    
    def _find_matching_words(self, pattern: str, guessed_letters: Set[str]) -> List[str]:
        """Find all words matching the current pattern."""
        length = len(pattern)
        candidates = self.words_by_length.get(length, [])
        
        matching = []
        for word in candidates:
            if self._word_matches_pattern(word, pattern, guessed_letters):
                matching.append(word)
        
        return matching
    
    def _word_matches_pattern(self, word: str, pattern: str, guessed_letters: Set[str]) -> bool:
        """Check if word matches pattern and constraints."""
        if len(word) != len(pattern):
            return False
        
        for w_char, p_char in zip(word, pattern):
            if p_char != '_':
                # Known position must match exactly
                if w_char != p_char:
                    return False
            else:
                # Unknown position cannot be a guessed letter
                if w_char in guessed_letters:
                    return False
        
        return True


class HangmanEnv:
    """Optimized Hangman environment."""
    
    def __init__(self, word: str, max_wrong: int = 6):
        self.target_word = word.lower()
        self.max_wrong = max_wrong
        self.reset()
    
    def reset(self):
        """Reset game state."""
        self.guessed_letters = set()
        self.wrong_guesses = 0
        self.repeated_guesses = 0
        self.correct_guesses = 0
        self.masked_word = '_' * len(self.target_word)
        return self.get_state()
    
    def get_state(self):
        """Get current state."""
        return (self.masked_word, self.guessed_letters.copy(), self.wrong_guesses)
    
    def step(self, letter: str):
        """Take action and return (state, reward, done)."""
        letter = letter.lower()
        
        # Penalty for repeated guess
        if letter in self.guessed_letters:
            self.repeated_guesses += 1
            return (self.get_state(), -10, False)
        
        self.guessed_letters.add(letter)
        
        # Check if letter is in word
        if letter in self.target_word:
            # Count occurrences
            count = self.target_word.count(letter)
            self.correct_guesses += count
            
            # Update masked word
            new_masked = list(self.masked_word)
            for i, char in enumerate(self.target_word):
                if char == letter:
                    new_masked[i] = letter
            self.masked_word = ''.join(new_masked)
            
            # Reward proportional to letters revealed
            reward = 5 * count
            
            # Check if won
            if '_' not in self.masked_word:
                reward += 100  # Big win bonus
                return (self.get_state(), reward, True)
            
            return (self.get_state(), reward, False)
        else:
            # Wrong guess
            self.wrong_guesses += 1
            reward = -15
            
            # Check if lost
            if self.wrong_guesses >= self.max_wrong:
                reward = -50  # Loss penalty
                return (self.get_state(), reward, True)
            
            return (self.get_state(), reward, False)
    
    def is_won(self) -> bool:
        """Check if game is won."""
        return '_' not in self.masked_word


class SmartAgent:
    """
    Smart agent that relies heavily on HMM probabilities.
    Uses simple greedy strategy with HMM guidance.
    """
    
    def __init__(self, hmm: ImprovedHMM):
        self.hmm = hmm
        self.stats = {'correct': 0, 'wrong': 0}
    
    def choose_action(self, state: Tuple[str, Set[str], int]) -> str:
        """Choose best letter based on HMM probabilities."""
        masked_word, guessed_letters, wrong_count = state
        
        # Get HMM probabilities
        probs = self.hmm.get_letter_probabilities(masked_word, guessed_letters)
        
        if not probs:
            # Fallback to most common letters not guessed
            common = 'etaoinshrdlcumwfgypbvkjxqz'
            for letter in common:
                if letter not in guessed_letters:
                    return letter
            return 'e'
        
        # Choose letter with highest probability
        return max(probs.items(), key=lambda x: x[1])[0]
    
    def update_stats(self, correct: bool):
        """Update statistics."""
        if correct:
            self.stats['correct'] += 1
        else:
            self.stats['wrong'] += 1


def play_games(agent: SmartAgent, words: List[str], verbose: bool = False) -> Tuple[int, int, int]:
    """
    Play multiple games and return statistics.
    Returns: (wins, total_wrong, total_repeated)
    """
    wins = 0
    total_wrong = 0
    total_repeated = 0
    
    for i, word in enumerate(words):
        env = HangmanEnv(word)
        state = env.reset()
        done = False
        
        while not done:
            action = agent.choose_action(state)
            next_state, reward, done = env.step(action)
            state = next_state
        
        if env.is_won():
            wins += 1
        
        total_wrong += env.wrong_guesses
        total_repeated += env.repeated_guesses
        
        if verbose and (i + 1) % 500 == 0:
            success_rate = wins / (i + 1)
            print(f"  Games {i+1}: Success Rate = {success_rate*100:.2f}%")
    
    return wins, total_wrong, total_repeated


def evaluate(agent: SmartAgent, test_file: str) -> Dict:
    """Evaluate agent on test set."""
    print("\nEvaluating on test set...")
    
    with open(test_file, 'r') as f:
        test_words = [line.strip().lower() for line in f if line.strip()]
    
    wins, total_wrong, total_repeated = play_games(agent, test_words, verbose=True)
    
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
        'avg_wrong_guesses': avg_wrong,
        'avg_repeated_guesses': avg_repeated,
        'total_wrong': total_wrong,
        'total_repeated': total_repeated,
        'final_score': final_score
    }


def main():
    """Main execution."""
    print("=" * 70)
    print("Hangman Agent - Improved HMM Strategy (Version 2)")
    print("=" * 70)
    
    corpus_file = 'data/corpus.txt'
    test_file = 'data/test.txt'
    
    # Train HMM
    hmm = ImprovedHMM()
    hmm.train(corpus_file)
    
    # Create agent
    agent = SmartAgent(hmm)
    
    # Evaluate
    results = evaluate(agent, test_file)
    
    # Print results
    print("\n" + "=" * 70)
    print("EVALUATION RESULTS - VERSION 2")
    print("=" * 70)
    print(f"Success Rate: {results['success_rate']*100:.2f}% ({results['wins']}/{results['total_games']})")
    print(f"Average Wrong Guesses: {results['avg_wrong_guesses']:.2f}")
    print(f"Average Repeated Guesses: {results['avg_repeated_guesses']:.2f}")
    print(f"Total Wrong Guesses: {results['total_wrong']}")
    print(f"Total Repeated Guesses: {results['total_repeated']}")
    print(f"\nFINAL SCORE: {results['final_score']:.2f}")
    print("=" * 70)
    
    return results


if __name__ == "__main__":
    main()
