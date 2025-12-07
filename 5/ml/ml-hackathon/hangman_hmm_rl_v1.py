"""
Hangman Agent using Hidden Markov Model + Reinforcement Learning
Version 1: Initial Implementation
"""

import numpy as np
import random
from collections import defaultdict, Counter
import pickle
import matplotlib.pyplot as plt
from typing import List, Tuple, Set, Dict


class HMMLetterPredictor:
    """
    Hidden Markov Model for predicting letter probabilities in Hangman.
    
    States: Letter positions in words
    Emissions: Actual letters
    """
    
    def __init__(self):
        self.word_patterns = defaultdict(lambda: defaultdict(int))
        self.letter_frequencies = defaultdict(int)
        self.length_words = defaultdict(list)
        
    def train(self, corpus_file: str):
        """Train HMM on corpus of words."""
        print("Training HMM on corpus...")
        
        with open(corpus_file, 'r') as f:
            words = [line.strip().lower() for line in f if line.strip()]
        
        # Group words by length
        for word in words:
            length = len(word)
            self.length_words[length].append(word)
            
            # Count letter frequencies at each position
            for pos, letter in enumerate(word):
                self.word_patterns[length][letter] += 1
                self.letter_frequencies[letter] += 1
        
        print(f"Trained on {len(words)} words")
        print(f"Word lengths: {sorted(self.length_words.keys())}")
        
    def get_letter_probabilities(self, masked_word: str, guessed_letters: Set[str]) -> Dict[str, float]:
        """
        Get probability distribution over remaining letters.
        
        Args:
            masked_word: Current state like "__pp__"
            guessed_letters: Set of already guessed letters
        
        Returns:
            Dictionary of letter -> probability
        """
        length = len(masked_word)
        remaining_letters = set('abcdefghijklmnopqrstuvwxyz') - guessed_letters
        
        # Find matching words from corpus
        matching_words = []
        for word in self.length_words.get(length, []):
            if self._matches_pattern(word, masked_word, guessed_letters):
                matching_words.append(word)
        
        if not matching_words:
            # Fallback to general letter frequencies
            letter_counts = {letter: self.letter_frequencies.get(letter, 1) 
                           for letter in remaining_letters}
        else:
            # Count letters in matching words
            letter_counts = defaultdict(int)
            for word in matching_words:
                for letter in word:
                    if letter in remaining_letters:
                        letter_counts[letter] += 1
        
        # Normalize to probabilities
        total = sum(letter_counts.values())
        if total == 0:
            # Uniform distribution
            probs = {letter: 1.0 / len(remaining_letters) for letter in remaining_letters}
        else:
            probs = {letter: count / total for letter, count in letter_counts.items()}
        
        return probs
    
    def _matches_pattern(self, word: str, pattern: str, guessed_letters: Set[str]) -> bool:
        """Check if word matches the current pattern."""
        if len(word) != len(pattern):
            return False
        
        for w_char, p_char in zip(word, pattern):
            if p_char != '_':
                # Known position must match
                if w_char != p_char:
                    return False
            else:
                # Unknown position must not be a guessed letter
                if w_char in guessed_letters:
                    return False
        
        return True


class HangmanEnvironment:
    """Hangman game environment for RL agent."""
    
    def __init__(self, word: str, max_wrong: int = 6):
        self.target_word = word.lower()
        self.max_wrong = max_wrong
        self.reset()
    
    def reset(self):
        """Reset game state."""
        self.guessed_letters = set()
        self.wrong_guesses = 0
        self.repeated_guesses = 0
        self.masked_word = '_' * len(self.target_word)
        return self.get_state()
    
    def get_state(self) -> Tuple[str, Set[str], int]:
        """Get current state."""
        return (self.masked_word, self.guessed_letters.copy(), self.wrong_guesses)
    
    def step(self, letter: str) -> Tuple[Tuple, float, bool]:
        """
        Take action (guess letter).
        
        Returns:
            (new_state, reward, done)
        """
        letter = letter.lower()
        
        # Check for repeated guess
        if letter in self.guessed_letters:
            self.repeated_guesses += 1
            reward = -5  # Heavy penalty for repeated guess
            return (self.get_state(), reward, False)
        
        self.guessed_letters.add(letter)
        
        # Check if letter is in word
        if letter in self.target_word:
            # Correct guess - update masked word
            new_masked = list(self.masked_word)
            for i, char in enumerate(self.target_word):
                if char == letter:
                    new_masked[i] = letter
            self.masked_word = ''.join(new_masked)
            
            # Reward based on progress
            reward = 2  # Positive reward for correct guess
            
            # Check if word is complete
            if '_' not in self.masked_word:
                reward = 50  # Big reward for winning
                return (self.get_state(), reward, True)
            
            return (self.get_state(), reward, False)
        else:
            # Wrong guess
            self.wrong_guesses += 1
            reward = -10  # Penalty for wrong guess
            
            # Check if game over
            if self.wrong_guesses >= self.max_wrong:
                reward = -100  # Big penalty for losing
                return (self.get_state(), reward, True)
            
            return (self.get_state(), reward, False)
    
    def is_done(self) -> bool:
        """Check if game is over."""
        return '_' not in self.masked_word or self.wrong_guesses >= self.max_wrong
    
    def is_won(self) -> bool:
        """Check if game is won."""
        return '_' not in self.masked_word


class QLearningAgent:
    """Q-Learning agent for Hangman."""
    
    def __init__(self, hmm: HMMLetterPredictor, alpha: float = 0.1, 
                 gamma: float = 0.95, epsilon: float = 1.0, epsilon_decay: float = 0.995):
        self.hmm = hmm
        self.alpha = alpha  # Learning rate
        self.gamma = gamma  # Discount factor
        self.epsilon = epsilon  # Exploration rate
        self.epsilon_decay = epsilon_decay
        self.epsilon_min = 0.01
        self.q_table = defaultdict(lambda: defaultdict(float))
    
    def get_state_key(self, state: Tuple[str, Set[str], int]) -> str:
        """Convert state to hashable key."""
        masked_word, guessed_letters, wrong_count = state
        guessed_str = ''.join(sorted(guessed_letters))
        return f"{masked_word}|{guessed_str}|{wrong_count}"
    
    def choose_action(self, state: Tuple[str, Set[str], int]) -> str:
        """Choose action using epsilon-greedy policy with HMM guidance."""
        masked_word, guessed_letters, wrong_count = state
        remaining_letters = set('abcdefghijklmnopqrstuvwxyz') - guessed_letters
        
        if not remaining_letters:
            return random.choice(list('abcdefghijklmnopqrstuvwxyz'))
        
        # Get HMM probabilities
        hmm_probs = self.hmm.get_letter_probabilities(masked_word, guessed_letters)
        
        # Epsilon-greedy exploration
        if random.random() < self.epsilon:
            # Explore: use HMM probabilities for guided exploration
            if hmm_probs:
                letters = list(hmm_probs.keys())
                probs = list(hmm_probs.values())
                if sum(probs) > 0:
                    probs = np.array(probs) / sum(probs)
                    return np.random.choice(letters, p=probs)
            return random.choice(list(remaining_letters))
        else:
            # Exploit: use Q-values weighted by HMM probabilities
            state_key = self.get_state_key(state)
            q_values = {}
            
            for letter in remaining_letters:
                q_val = self.q_table[state_key][letter]
                hmm_prob = hmm_probs.get(letter, 0.01)
                # Combine Q-value with HMM probability
                q_values[letter] = q_val + 10 * hmm_prob  # Weight HMM heavily
            
            return max(q_values.items(), key=lambda x: x[1])[0]
    
    def update(self, state: Tuple, action: str, reward: float, next_state: Tuple, done: bool):
        """Update Q-table."""
        state_key = self.get_state_key(state)
        next_state_key = self.get_state_key(next_state)
        
        # Current Q-value
        current_q = self.q_table[state_key][action]
        
        # Maximum Q-value for next state
        if done:
            max_next_q = 0
        else:
            next_masked, next_guessed, _ = next_state
            remaining = set('abcdefghijklmnopqrstuvwxyz') - next_guessed
            if remaining:
                max_next_q = max([self.q_table[next_state_key][a] for a in remaining], default=0)
            else:
                max_next_q = 0
        
        # Q-learning update
        new_q = current_q + self.alpha * (reward + self.gamma * max_next_q - current_q)
        self.q_table[state_key][action] = new_q
    
    def decay_epsilon(self):
        """Decay exploration rate."""
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)


def train_agent(agent: QLearningAgent, words: List[str], episodes: int = 5000) -> List[float]:
    """Train RL agent on corpus words."""
    print(f"\nTraining RL agent for {episodes} episodes...")
    
    episode_rewards = []
    
    for episode in range(episodes):
        # Select random word
        word = random.choice(words)
        env = HangmanEnvironment(word)
        state = env.reset()
        
        total_reward = 0
        done = False
        
        while not done:
            action = agent.choose_action(state)
            next_state, reward, done = env.step(action)
            agent.update(state, action, reward, next_state, done)
            
            state = next_state
            total_reward += reward
        
        episode_rewards.append(total_reward)
        agent.decay_epsilon()
        
        if (episode + 1) % 500 == 0:
            avg_reward = np.mean(episode_rewards[-500:])
            print(f"Episode {episode + 1}/{episodes}, Avg Reward: {avg_reward:.2f}, Epsilon: {agent.epsilon:.3f}")
    
    return episode_rewards


def evaluate_agent(agent: QLearningAgent, test_file: str) -> Dict:
    """Evaluate agent on test set."""
    print("\nEvaluating agent on test set...")
    
    with open(test_file, 'r') as f:
        test_words = [line.strip().lower() for line in f if line.strip()]
    
    wins = 0
    total_wrong = 0
    total_repeated = 0
    
    for word in test_words:
        env = HangmanEnvironment(word)
        state = env.reset()
        done = False
        
        # Use greedy policy (no exploration)
        old_epsilon = agent.epsilon
        agent.epsilon = 0
        
        while not done:
            action = agent.choose_action(state)
            next_state, reward, done = env.step(action)
            state = next_state
        
        agent.epsilon = old_epsilon
        
        if env.is_won():
            wins += 1
        
        total_wrong += env.wrong_guesses
        total_repeated += env.repeated_guesses
    
    success_rate = wins / len(test_words)
    avg_wrong = total_wrong / len(test_words)
    avg_repeated = total_repeated / len(test_words)
    
    # Calculate final score
    final_score = (success_rate * 2000) - (total_wrong * 5) - (total_repeated * 2)
    
    results = {
        'success_rate': success_rate,
        'wins': wins,
        'total_games': len(test_words),
        'avg_wrong_guesses': avg_wrong,
        'avg_repeated_guesses': avg_repeated,
        'total_wrong': total_wrong,
        'total_repeated': total_repeated,
        'final_score': final_score
    }
    
    return results


def main():
    """Main execution."""
    print("=" * 60)
    print("Hangman Agent - HMM + RL (Version 1)")
    print("=" * 60)
    
    # Paths
    corpus_file = 'data/corpus.txt'
    test_file = 'data/test.txt'
    
    # Train HMM
    hmm = HMMLetterPredictor()
    hmm.train(corpus_file)
    
    # Load corpus words for RL training
    with open(corpus_file, 'r') as f:
        corpus_words = [line.strip().lower() for line in f if line.strip()]
    
    # Create and train RL agent
    agent = QLearningAgent(hmm, alpha=0.1, gamma=0.95, epsilon=1.0, epsilon_decay=0.995)
    episode_rewards = train_agent(agent, corpus_words, episodes=10000)
    
    # Plot training progress
    plt.figure(figsize=(10, 5))
    plt.plot(episode_rewards, alpha=0.3)
    window = 100
    moving_avg = np.convolve(episode_rewards, np.ones(window)/window, mode='valid')
    plt.plot(range(window-1, len(episode_rewards)), moving_avg, linewidth=2)
    plt.xlabel('Episode')
    plt.ylabel('Total Reward')
    plt.title('Training Progress')
    plt.grid(True)
    plt.savefig('training_progress_v1.png')
    print("\nSaved training progress plot to training_progress_v1.png")
    
    # Evaluate on test set
    results = evaluate_agent(agent, test_file)
    
    # Print results
    print("\n" + "=" * 60)
    print("EVALUATION RESULTS")
    print("=" * 60)
    print(f"Success Rate: {results['success_rate']*100:.2f}% ({results['wins']}/{results['total_games']})")
    print(f"Average Wrong Guesses: {results['avg_wrong_guesses']:.2f}")
    print(f"Average Repeated Guesses: {results['avg_repeated_guesses']:.2f}")
    print(f"Total Wrong Guesses: {results['total_wrong']}")
    print(f"Total Repeated Guesses: {results['total_repeated']}")
    print(f"\nFINAL SCORE: {results['final_score']:.2f}")
    print("=" * 60)
    
    # Save agent
    with open('agent_v1.pkl', 'wb') as f:
        pickle.dump({'hmm': hmm, 'agent': agent, 'results': results}, f)
    print("\nSaved agent to agent_v1.pkl")
    
    return results


if __name__ == "__main__":
    main()
