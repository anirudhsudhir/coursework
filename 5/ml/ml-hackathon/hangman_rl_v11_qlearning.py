"""
Hangman Agent - Q-Learning with Feature-Based State Representation
Version 11: Proper RL with Q-Learning and Feature Approximation

Key Improvements:
1. Feature-based state representation (solves state space explosion)
2. Q-Learning with epsilon-greedy exploration
3. Experience replay for better sample efficiency
4. Gradually decaying exploration rate
5. Learns from rewards during training
"""

import numpy as np
from collections import defaultdict, Counter, deque
from typing import List, Tuple, Set, Dict
import random
import pickle


class FeatureExtractor:
    """Extract features from game state to reduce dimensionality."""
    
    def __init__(self):
        self.alphabet = 'abcdefghijklmnopqrstuvwxyz'
    
    def extract_features(self, pattern: str, guessed: Set[str], wrong_count: int) -> np.ndarray:
        """
        Extract feature vector from state.
        
        Features:
        - Word length (normalized)
        - Number of blanks remaining (normalized)
        - Number of letters guessed (normalized)
        - Wrong count (normalized)
        - Letter frequency features (26 binary: guessed or not)
        - Pattern position features (first/last letter revealed)
        """
        length = len(pattern)
        blanks = pattern.count('_')
        num_guessed = len(guessed)
        
        features = []
        
        # Basic features (normalized to [0, 1])
        features.append(length / 24.0)  # Max length 24
        features.append(blanks / max(length, 1))  # Proportion of blanks
        features.append(num_guessed / 26.0)  # Proportion guessed
        features.append(wrong_count / 6.0)  # Normalized wrong count
        
        # Letter guessed binary features (26 features)
        for letter in self.alphabet:
            features.append(1.0 if letter in guessed else 0.0)
        
        # Pattern structure features
        features.append(1.0 if pattern[0] != '_' else 0.0)  # First letter revealed
        features.append(1.0 if pattern[-1] != '_' else 0.0)  # Last letter revealed
        
        # Revealed letter count at start/middle/end
        third = max(length // 3, 1)
        start_revealed = sum(1 for c in pattern[:third] if c != '_') / third
        middle_revealed = sum(1 for c in pattern[third:2*third] if c != '_') / third if length > 3 else 0
        end_revealed = sum(1 for c in pattern[-third:] if c != '_') / third
        
        features.extend([start_revealed, middle_revealed, end_revealed])
        
        return np.array(features, dtype=np.float32)


class QLearningAgent:
    """Q-Learning agent with linear function approximation."""
    
    def __init__(self, learning_rate: float = 0.01, discount: float = 0.95, 
                 epsilon_start: float = 1.0, epsilon_end: float = 0.05, 
                 epsilon_decay: float = 0.995):
        self.alphabet = 'abcdefghijklmnopqrstuvwxyz'
        self.letter_to_idx = {letter: i for i, letter in enumerate(self.alphabet)}
        
        self.feature_extractor = FeatureExtractor()
        
        # Determine feature dimension from a sample
        sample_features = self.feature_extractor.extract_features("___", set(), 0)
        self.feature_dim = len(sample_features)
        
        # Q-function weights: one weight vector per action (letter)
        self.weights = np.zeros((26, self.feature_dim), dtype=np.float32)
        
        # Hyperparameters
        self.learning_rate = learning_rate
        self.discount = discount
        self.epsilon = epsilon_start
        self.epsilon_end = epsilon_end
        self.epsilon_decay = epsilon_decay
        
        # Experience replay buffer
        self.replay_buffer = deque(maxlen=10000)
        self.batch_size = 32
        
        # Training statistics
        self.episode_rewards = []
        self.episode_lengths = []
        
    def get_q_value(self, state_features: np.ndarray, action: str) -> float:
        """Compute Q(s, a) = w_a^T * features(s)"""
        action_idx = self.letter_to_idx[action]
        return np.dot(self.weights[action_idx], state_features)
    
    def get_action(self, pattern: str, guessed: Set[str], wrong_count: int, 
                   training: bool = True) -> str:
        """
        Choose action using epsilon-greedy policy.
        
        Args:
            pattern: Current word pattern
            guessed: Set of guessed letters
            wrong_count: Number of wrong guesses
            training: If True, use epsilon-greedy; if False, use greedy
        """
        available_letters = set(self.alphabet) - guessed
        
        if not available_letters:
            return random.choice(list(self.alphabet))
        
        # Epsilon-greedy exploration during training
        if training and random.random() < self.epsilon:
            return random.choice(list(available_letters))
        
        # Greedy: choose action with highest Q-value
        state_features = self.feature_extractor.extract_features(pattern, guessed, wrong_count)
        
        best_action = None
        best_q = float('-inf')
        
        for action in available_letters:
            q_val = self.get_q_value(state_features, action)
            if q_val > best_q:
                best_q = q_val
                best_action = action
        
        return best_action
    
    def update(self, state: Tuple, action: str, reward: float, 
               next_state: Tuple, done: bool):
        """
        Q-Learning update: Q(s,a) ← Q(s,a) + α[r + γ max_a' Q(s',a') - Q(s,a)]
        """
        pattern, guessed, wrong_count = state
        next_pattern, next_guessed, next_wrong_count = next_state
        
        # Current Q-value
        state_features = self.feature_extractor.extract_features(pattern, guessed, wrong_count)
        current_q = self.get_q_value(state_features, action)
        
        # Target Q-value
        if done:
            target_q = reward  # No future rewards
        else:
            # Max Q-value over next actions
            next_state_features = self.feature_extractor.extract_features(
                next_pattern, next_guessed, next_wrong_count
            )
            available_next = set(self.alphabet) - next_guessed
            
            max_next_q = float('-inf')
            for next_action in available_next:
                next_q = self.get_q_value(next_state_features, next_action)
                max_next_q = max(max_next_q, next_q)
            
            target_q = reward + self.discount * max_next_q
        
        # Gradient descent update
        td_error = target_q - current_q
        action_idx = self.letter_to_idx[action]
        self.weights[action_idx] += self.learning_rate * td_error * state_features
        
        # Store experience for replay
        self.replay_buffer.append((state, action, reward, next_state, done))
    
    def experience_replay(self):
        """Sample from replay buffer and perform additional updates."""
        if len(self.replay_buffer) < self.batch_size:
            return
        
        # Sample random batch
        batch = random.sample(self.replay_buffer, self.batch_size)
        
        for state, action, reward, next_state, done in batch:
            self.update(state, action, reward, next_state, done)
    
    def decay_epsilon(self):
        """Decay exploration rate."""
        self.epsilon = max(self.epsilon_end, self.epsilon * self.epsilon_decay)
    
    def save(self, filepath: str):
        """Save model weights."""
        data = {
            'weights': self.weights,
            'epsilon': self.epsilon,
            'episode_rewards': self.episode_rewards,
            'episode_lengths': self.episode_lengths
        }
        with open(filepath, 'wb') as f:
            pickle.dump(data, f)
        print(f"Model saved to {filepath}")
    
    def load(self, filepath: str):
        """Load model weights."""
        with open(filepath, 'rb') as f:
            data = pickle.load(f)
        self.weights = data['weights']
        self.epsilon = data.get('epsilon', self.epsilon_end)
        self.episode_rewards = data.get('episode_rewards', [])
        self.episode_lengths = data.get('episode_lengths', [])
        print(f"Model loaded from {filepath}")


class HangmanGame:
    """Hangman game environment."""
    
    def __init__(self, word: str, max_wrong: int = 6):
        self.word = word.lower()
        self.max_wrong = max_wrong
        self.reset()
    
    def reset(self):
        """Reset game state."""
        self.pattern = ['_'] * len(self.word)
        self.guessed = set()
        self.wrong = 0
        return self._get_state()
    
    def _get_state(self) -> Tuple[str, Set[str], int]:
        """Return current state."""
        return (''.join(self.pattern), self.guessed.copy(), self.wrong)
    
    def step(self, letter: str) -> Tuple[Tuple, float, bool]:
        """
        Take action and return (next_state, reward, done).
        
        Reward structure:
        - Correct guess: +10
        - Wrong guess: -20
        - Win: +100
        - Loss: -100
        """
        letter = letter.lower()
        self.guessed.add(letter)
        
        if letter in self.word:
            # Correct guess - reveal letters
            for i, char in enumerate(self.word):
                if char == letter:
                    self.pattern[i] = letter
            reward = 10
        else:
            # Wrong guess
            self.wrong += 1
            reward = -20
        
        # Check terminal state
        done = self.is_won() or self.is_lost()
        
        if done:
            if self.is_won():
                reward += 100  # Bonus for winning
            else:
                reward -= 100  # Penalty for losing
        
        return self._get_state(), reward, done
    
    def is_won(self) -> bool:
        """Check if game is won."""
        return '_' not in self.pattern
    
    def is_lost(self) -> bool:
        """Check if game is lost."""
        return self.wrong >= self.max_wrong
    
    def is_done(self) -> bool:
        """Check if game is over."""
        return self.is_won() or self.is_lost()


def train_agent(agent: QLearningAgent, words: List[str], num_episodes: int = 5000):
    """Train Q-Learning agent."""
    print(f"\nTraining Q-Learning Agent for {num_episodes} episodes...")
    
    for episode in range(num_episodes):
        # Sample random word
        word = random.choice(words)
        game = HangmanGame(word)
        
        state = game.reset()
        total_reward = 0
        steps = 0
        
        while not game.is_done():
            # Choose action
            pattern, guessed, wrong_count = state
            action = agent.get_action(pattern, guessed, wrong_count, training=True)
            
            # Take action
            next_state, reward, done = game.step(action)
            
            # Update Q-values
            agent.update(state, action, reward, next_state, done)
            
            total_reward += reward
            steps += 1
            state = next_state
        
        # Experience replay for better learning
        if episode % 10 == 0:
            agent.experience_replay()
        
        # Decay exploration
        agent.decay_epsilon()
        
        # Track statistics
        agent.episode_rewards.append(total_reward)
        agent.episode_lengths.append(steps)
        
        # Print progress
        if (episode + 1) % 100 == 0:
            avg_reward = np.mean(agent.episode_rewards[-100:])
            avg_length = np.mean(agent.episode_lengths[-100:])
            print(f"Episode {episode + 1}/{num_episodes} | "
                  f"Avg Reward: {avg_reward:.2f} | "
                  f"Avg Length: {avg_length:.2f} | "
                  f"Epsilon: {agent.epsilon:.4f}")
    
    print("Training completed!")
    return agent


def evaluate_agent(agent: QLearningAgent, test_words: List[str]) -> Dict:
    """Evaluate trained agent on test set."""
    print(f"\nEvaluating on {len(test_words)} test words...")
    
    wins = 0
    total_wrong = 0
    total_repeated = 0
    
    for i, word in enumerate(test_words):
        game = HangmanGame(word)
        state = game.reset()
        
        while not game.is_done():
            pattern, guessed, wrong_count = state
            action = agent.get_action(pattern, guessed, wrong_count, training=False)
            
            # Check for repeated guess
            if action in guessed:
                total_repeated += 1
            
            state, _, _ = game.step(action)
        
        if game.is_won():
            wins += 1
        total_wrong += game.wrong
        
        if (i + 1) % 100 == 0:
            print(f"Evaluated {i + 1}/{len(test_words)} words...")
    
    success_rate = wins / len(test_words)
    avg_wrong = total_wrong / len(test_words)
    avg_repeated = total_repeated / len(test_words)
    
    # Calculate final score
    final_score = (success_rate * 2000) - (total_wrong * 5) - (total_repeated * 2)
    
    results = {
        'success_rate': success_rate,
        'wins': wins,
        'total_wrong': total_wrong,
        'avg_wrong': avg_wrong,
        'total_repeated': total_repeated,
        'avg_repeated': avg_repeated,
        'final_score': final_score
    }
    
    return results


def main():
    """Main execution."""
    print("=" * 60)
    print("Hangman Q-Learning Agent - Version 11")
    print("=" * 60)
    
    # Load data
    corpus_file = 'data/corpus.txt'
    test_file = 'data/test.txt'
    
    print(f"\nLoading training data from {corpus_file}...")
    with open(corpus_file, 'r') as f:
        train_words = [line.strip().lower() for line in f if line.strip()]
    print(f"Loaded {len(train_words)} training words")
    
    print(f"\nLoading test data from {test_file}...")
    with open(test_file, 'r') as f:
        test_words = [line.strip().lower() for line in f if line.strip()]
    print(f"Loaded {len(test_words)} test words")
    
    # Initialize agent
    agent = QLearningAgent(
        learning_rate=0.01,
        discount=0.95,
        epsilon_start=1.0,
        epsilon_end=0.05,
        epsilon_decay=0.995
    )
    
    # Train agent
    agent = train_agent(agent, train_words, num_episodes=5000)
    
    # Save trained model
    agent.save('models/qlearning_v11.pkl')
    
    # Evaluate on test set
    results = evaluate_agent(agent, test_words)
    
    # Print results
    print("\n" + "=" * 60)
    print("FINAL RESULTS")
    print("=" * 60)
    print(f"Success Rate: {results['success_rate']:.4f} ({results['wins']}/{len(test_words)})")
    print(f"Average Wrong Guesses: {results['avg_wrong']:.4f}")
    print(f"Average Repeated Guesses: {results['avg_repeated']:.4f}")
    print(f"Total Wrong Guesses: {results['total_wrong']}")
    print(f"Total Repeated Guesses: {results['total_repeated']}")
    print(f"\nFinal Score: {results['final_score']:.2f}")
    print("=" * 60)


if __name__ == "__main__":
    main()
