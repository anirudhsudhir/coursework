"""
Hangman Agent - Deep Q-Network with HMM Guidance
Version 16: DQN with 70:30 HMM-to-Learned Ratio

Key Features:
1. Neural network Q-function approximation
2. Experience replay buffer
3. Target network for stable learning
4. 70% HMM priors + 30% learned DQN weights
5. Minimal exploration to maintain accuracy
"""

import numpy as np
from collections import defaultdict, Counter, deque
from typing import List, Tuple, Set, Dict
import random


class SimpleDQN:
    """Simple Deep Q-Network using linear approximation."""
    
    def __init__(self, input_dim: int, output_dim: int = 26):
        self.input_dim = input_dim
        self.output_dim = output_dim
        
        # Network weights (simple linear model)
        self.W1 = np.random.randn(input_dim, 64) * 0.01
        self.b1 = np.zeros(64)
        self.W2 = np.random.randn(64, output_dim) * 0.01
        self.b2 = np.zeros(output_dim)
        
        # Target network (for stability)
        self.target_W1 = self.W1.copy()
        self.target_b1 = self.b1.copy()
        self.target_W2 = self.W2.copy()
        self.target_b2 = self.b2.copy()
    
    def forward(self, state: np.ndarray, use_target: bool = False) -> np.ndarray:
        """Forward pass through network."""
        W1 = self.target_W1 if use_target else self.W1
        b1 = self.target_b1 if use_target else self.b1
        W2 = self.target_W2 if use_target else self.W2
        b2 = self.target_b2 if use_target else self.b2
        
        # Hidden layer with ReLU
        h = np.maximum(0, np.dot(state, W1) + b1)
        # Output layer
        q_values = np.dot(h, W2) + b2
        return q_values
    
    def update(self, state: np.ndarray, action: int, target: float, learning_rate: float = 0.001):
        """Update network weights using gradient descent."""
        # Forward pass
        h = np.maximum(0, np.dot(state, self.W1) + self.b1)
        q_values = np.dot(h, self.W2) + self.b2
        
        # Compute gradients
        dq = np.zeros(self.output_dim)
        dq[action] = 2 * (q_values[action] - target)
        
        # Backprop
        dW2 = np.outer(h, dq)
        db2 = dq
        
        dh = np.dot(dq, self.W2.T)
        dh[h <= 0] = 0  # ReLU gradient
        
        dW1 = np.outer(state, dh)
        db1 = dh
        
        # Update weights
        self.W2 -= learning_rate * dW2
        self.b2 -= learning_rate * db2
        self.W1 -= learning_rate * dW1
        self.b1 -= learning_rate * db1
    
    def update_target(self):
        """Copy weights to target network."""
        self.target_W1 = self.W1.copy()
        self.target_b1 = self.b1.copy()
        self.target_W2 = self.W2.copy()
        self.target_b2 = self.b2.copy()


class HMMPriorModel:
    """HMM model for providing prior probabilities."""
    
    def __init__(self):
        self.alphabet = 'abcdefghijklmnopqrstuvwxyz'
        self.letter_probs = {}
        self.bigram_probs = defaultdict(lambda: defaultdict(float))
        self.trigram_probs = defaultdict(lambda: defaultdict(lambda: defaultdict(float)))
        self.position_probs = defaultdict(lambda: defaultdict(lambda: defaultdict(float)))
    
    def train(self, words: List[str]):
        """Train HMM priors."""
        print("Training HMM priors...")
        
        # Unigrams
        letter_counts = Counter()
        for word in words:
            letter_counts.update(word)
        total = sum(letter_counts.values())
        self.letter_probs = {l: (letter_counts[l] + 0.5) / (total + 13) for l in self.alphabet}
        
        # Bigrams
        bigram_counts = defaultdict(Counter)
        for word in words:
            for i in range(len(word) - 1):
                bigram_counts[word[i]][word[i+1]] += 1
        
        for l1 in self.alphabet:
            total = sum(bigram_counts[l1].values()) + 13
            for l2 in self.alphabet:
                self.bigram_probs[l1][l2] = (bigram_counts[l1][l2] + 0.5) / total
        
        # Trigrams
        trigram_counts = defaultdict(lambda: defaultdict(Counter))
        for word in words:
            for i in range(len(word) - 2):
                trigram_counts[word[i]][word[i+1]][word[i+2]] += 1
        
        for l1 in self.alphabet:
            for l2 in self.alphabet:
                total = sum(trigram_counts[l1][l2].values()) + 13
                for l3 in self.alphabet:
                    self.trigram_probs[l1][l2][l3] = (trigram_counts[l1][l2][l3] + 0.5) / total
        
        # Position-based
        position_counts = defaultdict(lambda: defaultdict(Counter))
        for word in words:
            length = len(word)
            for pos, letter in enumerate(word):
                position_counts[length][pos][letter] += 1
        
        for length in range(1, 25):
            if length in position_counts:
                for pos in range(length):
                    total = sum(position_counts[length][pos].values()) + 13
                    for letter in self.alphabet:
                        self.position_probs[length][pos][letter] = \
                            (position_counts[length][pos][letter] + 0.5) / total
    
    def get_probabilities(self, pattern: str, guessed: Set[str]) -> Dict[str, float]:
        """Get HMM probability scores."""
        available = set(self.alphabet) - guessed
        scores = defaultdict(float)
        
        # Position-based
        length = len(pattern)
        if length in self.position_probs:
            for i, char in enumerate(pattern):
                if char == '_':
                    for letter in available:
                        scores[letter] += self.position_probs[length][i].get(letter, 0) * 3.0
        
        # Context
        revealed = [c for c in pattern if c != '_']
        
        for letter in available:
            scores[letter] += self.letter_probs.get(letter, 0) * 1.0
        
        if len(revealed) >= 1:
            last = revealed[-1]
            for letter in available:
                scores[letter] += self.bigram_probs[last][letter] * 2.0
        
        if len(revealed) >= 2:
            prev2, prev1 = revealed[-2], revealed[-1]
            for letter in available:
                scores[letter] += self.trigram_probs[prev2][prev1][letter] * 2.5
        
        # Normalize
        total = sum(scores.values())
        if total > 0:
            return {l: scores[l] / total for l in available}
        return {l: 1.0 / len(available) for l in available}


class DQNAgent:
    """DQN Agent with 70:30 HMM-to-Learned ratio."""
    
    def __init__(self, hmm_prior: HMMPriorModel,
                 learning_rate: float = 0.0005,
                 discount: float = 0.95,
                 epsilon_start: float = 0.05,
                 epsilon_end: float = 0.01,
                 epsilon_decay: float = 0.999,
                 hmm_weight: float = 0.7):
        
        self.alphabet = 'abcdefghijklmnopqrstuvwxyz'
        self.letter_to_idx = {l: i for i, l in enumerate(self.alphabet)}
        
        self.hmm_prior = hmm_prior
        self.hmm_weight = hmm_weight  # 70% HMM, 30% learned
        
        # DQN
        self.dqn = SimpleDQN(input_dim=35)  # State feature dimension
        
        # Hyperparameters
        self.learning_rate = learning_rate
        self.discount = discount
        self.epsilon = epsilon_start
        self.epsilon_end = epsilon_end
        self.epsilon_decay = epsilon_decay
        
        # Experience replay
        self.replay_buffer = deque(maxlen=5000)
        self.batch_size = 32
        
        # Statistics
        self.episode_rewards = []
        self.wins = []
        self.update_counter = 0
    
    def extract_features(self, pattern: str, guessed: Set[str], wrong: int) -> np.ndarray:
        """Extract state features."""
        length = len(pattern)
        blanks = pattern.count('_')
        num_guessed = len(guessed)
        
        features = [
            length / 24.0,
            blanks / max(length, 1),
            num_guessed / 26.0,
            wrong / 6.0
        ]
        
        # Guessed letters (26 binary features)
        for letter in self.alphabet:
            features.append(1.0 if letter in guessed else 0.0)
        
        # Pattern structure
        features.append(1.0 if pattern[0] != '_' else 0.0)
        features.append(1.0 if pattern[-1] != '_' else 0.0)
        
        # Revealed counts in segments
        third = max(length // 3, 1)
        start_revealed = sum(1 for c in pattern[:third] if c != '_') / third
        middle_revealed = sum(1 for c in pattern[third:2*third] if c != '_') / third if length > 3 else 0
        end_revealed = sum(1 for c in pattern[-third:] if c != '_') / third
        
        features.extend([start_revealed, middle_revealed, end_revealed])
        
        return np.array(features, dtype=np.float32)
    
    def get_action(self, pattern: str, guessed: Set[str], wrong: int, training: bool = True) -> str:
        """Select action using 70:30 HMM:DQN ratio with epsilon-greedy."""
        available = set(self.alphabet) - guessed
        if not available:
            return random.choice(list(self.alphabet))
        
        # Epsilon-greedy exploration
        if training and random.random() < self.epsilon:
            # Weighted random from HMM
            hmm_probs = self.hmm_prior.get_probabilities(pattern, guessed)
            letters = list(hmm_probs.keys())
            weights = [hmm_probs[l] for l in letters]
            return random.choices(letters, weights=weights)[0]
        
        # Get HMM priors
        hmm_probs = self.hmm_prior.get_probabilities(pattern, guessed)
        
        # Get DQN Q-values
        state_features = self.extract_features(pattern, guessed, wrong)
        q_values = self.dqn.forward(state_features)
        
        # Combine with 70:30 ratio
        combined_scores = {}
        for letter in available:
            idx = self.letter_to_idx[letter]
            hmm_score = hmm_probs.get(letter, 0)
            dqn_score = q_values[idx]
            
            # 70% HMM + 30% DQN
            combined_scores[letter] = self.hmm_weight * hmm_score + (1 - self.hmm_weight) * dqn_score
        
        return max(combined_scores.items(), key=lambda x: x[1])[0]
    
    def store_experience(self, state: Tuple, action: str, reward: float, next_state: Tuple, done: bool):
        """Store experience in replay buffer."""
        self.replay_buffer.append((state, action, reward, next_state, done))
    
    def train_step(self):
        """Perform one training step with experience replay."""
        if len(self.replay_buffer) < self.batch_size:
            return
        
        # Sample batch
        batch = random.sample(self.replay_buffer, self.batch_size)
        
        for state, action, reward, next_state, done in batch:
            pattern, guessed, wrong = state
            next_pattern, next_guessed, next_wrong = next_state
            
            # Current Q
            state_features = self.extract_features(pattern, guessed, wrong)
            action_idx = self.letter_to_idx[action]
            
            # Target Q
            if done:
                target = reward
            else:
                next_state_features = self.extract_features(next_pattern, next_guessed, next_wrong)
                next_q_values = self.dqn.forward(next_state_features, use_target=True)
                
                # Max over available actions
                next_available = set(self.alphabet) - next_guessed
                max_next_q = max([next_q_values[self.letter_to_idx[a]] for a in next_available], default=0)
                target = reward + self.discount * max_next_q
            
            # Update network
            self.dqn.update(state_features, action_idx, target, self.learning_rate)
        
        # Update target network periodically
        self.update_counter += 1
        if self.update_counter % 100 == 0:
            self.dqn.update_target()
    
    def decay_epsilon(self):
        """Decay exploration rate."""
        self.epsilon = max(self.epsilon_end, self.epsilon * self.epsilon_decay)


class HangmanGame:
    """Hangman game environment."""
    
    def __init__(self, word: str, max_wrong: int = 6):
        self.word = word.lower()
        self.max_wrong = max_wrong
        self.reset()
    
    def reset(self):
        self.pattern = ['_'] * len(self.word)
        self.guessed = set()
        self.wrong = 0
        return self._get_state()
    
    def _get_state(self) -> Tuple[str, Set[str], int]:
        return (''.join(self.pattern), self.guessed.copy(), self.wrong)
    
    def step(self, letter: str) -> Tuple[Tuple, float, bool]:
        letter = letter.lower()
        self.guessed.add(letter)
        
        if letter in self.word:
            num_revealed = sum(1 for char in self.word if char == letter)
            for i, char in enumerate(self.word):
                if char == letter:
                    self.pattern[i] = letter
            reward = 10 + (num_revealed * 3)
        else:
            self.wrong += 1
            reward = -20
        
        done = self.is_won() or self.is_lost()
        
        if done:
            if self.is_won():
                reward += 100
            else:
                reward -= 80
        
        return self._get_state(), reward, done
    
    def is_won(self) -> bool:
        return '_' not in self.pattern
    
    def is_lost(self) -> bool:
        return self.wrong >= self.max_wrong
    
    def is_done(self) -> bool:
        return self.is_won() or self.is_lost()


def train_dqn(agent: DQNAgent, words: List[str], num_episodes: int = 8000):
    """Train DQN agent."""
    print(f"\nTraining DQN Agent for {num_episodes} episodes...")
    print(f"HMM Weight: {agent.hmm_weight:.0%}, DQN Weight: {1-agent.hmm_weight:.0%}")
    
    for episode in range(num_episodes):
        word = random.choice(words)
        game = HangmanGame(word)
        
        state = game.reset()
        total_reward = 0
        steps = 0
        
        while not game.is_done():
            pattern, guessed, wrong = state
            action = agent.get_action(pattern, guessed, wrong, training=True)
            
            next_state, reward, done = game.step(action)
            
            # Store experience
            agent.store_experience(state, action, reward, next_state, done)
            
            # Train
            if episode % 5 == 0:  # Train every 5 episodes
                agent.train_step()
            
            total_reward += reward
            steps += 1
            state = next_state
        
        agent.decay_epsilon()
        agent.episode_rewards.append(total_reward)
        agent.wins.append(1 if game.is_won() else 0)
        
        if (episode + 1) % 500 == 0:
            recent_wins = sum(agent.wins[-500:])
            avg_reward = np.mean(agent.episode_rewards[-500:])
            print(f"Episode {episode + 1}/{num_episodes} | "
                  f"Win Rate: {recent_wins/500:.2%} | "
                  f"Avg Reward: {avg_reward:.2f} | "
                  f"Epsilon: {agent.epsilon:.4f}")
    
    print("Training completed!")
    return agent


def evaluate_agent(agent: DQNAgent, test_words: List[str]) -> Dict:
    """Evaluate agent on test set."""
    print(f"\nEvaluating on {len(test_words)} test words...")
    
    wins = 0
    total_wrong = 0
    total_repeated = 0
    
    for i, word in enumerate(test_words):
        game = HangmanGame(word)
        state = game.reset()
        
        while not game.is_done():
            pattern, guessed, wrong = state
            action = agent.get_action(pattern, guessed, wrong, training=False)
            
            if action in guessed:
                total_repeated += 1
            
            state, _, _ = game.step(action)
        
        if game.is_won():
            wins += 1
        total_wrong += game.wrong
        
        if (i + 1) % 200 == 0:
            print(f"Evaluated {i + 1}/{len(test_words)} words...")
    
    success_rate = wins / len(test_words)
    avg_wrong = total_wrong / len(test_words)
    avg_repeated = total_repeated / len(test_words)
    final_score = (success_rate * 2000) - (total_wrong * 5) - (total_repeated * 2)
    
    return {
        'success_rate': success_rate,
        'wins': wins,
        'total_wrong': total_wrong,
        'avg_wrong': avg_wrong,
        'total_repeated': total_repeated,
        'avg_repeated': avg_repeated,
        'final_score': final_score
    }


def main():
    print("=" * 70)
    print("Hangman DQN Agent - Version 16")
    print("Deep Q-Network with 70:30 HMM:Learned Ratio")
    print("=" * 70)
    
    # Load data
    corpus_file = 'data/corpus.txt'
    test_file = 'data/test.txt'
    
    with open(corpus_file, 'r') as f:
        train_words = [line.strip().lower() for line in f if line.strip()]
    print(f"Loaded {len(train_words)} training words")
    
    with open(test_file, 'r') as f:
        test_words = [line.strip().lower() for line in f if line.strip()]
    print(f"Loaded {len(test_words)} test words")
    
    # Train HMM priors
    hmm = HMMPriorModel()
    hmm.train(train_words)
    
    # Initialize DQN agent
    agent = DQNAgent(
        hmm_prior=hmm,
        learning_rate=0.0005,
        discount=0.95,
        epsilon_start=0.05,  # Low exploration
        epsilon_end=0.01,
        epsilon_decay=0.999,
        hmm_weight=0.7  # 70% HMM, 30% learned
    )
    
    # Train
    agent = train_dqn(agent, train_words, num_episodes=8000)
    
    # Evaluate
    results = evaluate_agent(agent, test_words)
    
    # Print results
    print("\n" + "=" * 70)
    print("FINAL RESULTS")
    print("=" * 70)
    print(f"Success Rate: {results['success_rate']:.4f} ({results['wins']}/{len(test_words)})")
    print(f"Average Wrong Guesses: {results['avg_wrong']:.4f}")
    print(f"Average Repeated Guesses: {results['avg_repeated']:.4f}")
    print(f"Total Wrong Guesses: {results['total_wrong']}")
    print(f"Total Repeated Guesses: {results['total_repeated']}")
    print(f"\nFinal Score: {results['final_score']:.2f}")
    print("=" * 70)


if __name__ == "__main__":
    main()
