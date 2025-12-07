"""
Hangman Agent - Policy Gradient with HMM Features
Version 14: Actor-Critic RL with HMM-Guided Policy

Key Features:
1. Use HMM probabilities as policy network initialization
2. Light exploration (5-10%) on top of HMM policy
3. Update policy weights based on rewards
4. Target: 30%+ success rate with TRUE RL learning
"""

import numpy as np
from collections import defaultdict, Counter
from typing import List, Tuple, Set, Dict
import random
import pickle


class EnhancedHMMModel:
    """Full HMM model like V8 for strong baseline."""
    
    def __init__(self):
        self.alphabet = 'abcdefghijklmnopqrstuvwxyz'
        self.letter_probs = {}
        self.bigram_probs = defaultdict(lambda: defaultdict(float))
        self.trigram_probs = defaultdict(lambda: defaultdict(lambda: defaultdict(float)))
        self.emission_probs = {}  # By length and position
    
    def train(self, words: List[str]):
        """Train comprehensive HMM model."""
        print("Training Enhanced HMM Model...")
        
        # Group by length
        words_by_length = defaultdict(list)
        for word in words:
            if len(word) <= 24:
                words_by_length[len(word)].append(word)
        
        # Train emission probabilities by length and position
        for length, length_words in words_by_length.items():
            self.emission_probs[length] = {}
            for pos in range(length):
                letter_counts = Counter()
                for word in length_words:
                    letter_counts[word[pos]] += 1
                
                total = sum(letter_counts.values())
                self.emission_probs[length][pos] = {}
                for letter in self.alphabet:
                    self.emission_probs[length][pos][letter] = \
                        (letter_counts[letter] + 0.5) / (total + 13)
        
        # Train unigrams
        all_letters = Counter()
        for word in words:
            all_letters.update(word)
        total = sum(all_letters.values())
        self.letter_probs = {letter: (all_letters[letter] + 0.5) / (total + 13) 
                            for letter in self.alphabet}
        
        # Train bigrams
        bigram_counts = defaultdict(Counter)
        for word in words:
            for i in range(len(word) - 1):
                bigram_counts[word[i]][word[i+1]] += 1
        
        for l1 in self.alphabet:
            total = sum(bigram_counts[l1].values()) + 13
            for l2 in self.alphabet:
                self.bigram_probs[l1][l2] = (bigram_counts[l1][l2] + 0.5) / total
        
        # Train trigrams
        trigram_counts = defaultdict(lambda: defaultdict(Counter))
        for word in words:
            for i in range(len(word) - 2):
                trigram_counts[word[i]][word[i+1]][word[i+2]] += 1
        
        for l1 in self.alphabet:
            for l2 in self.alphabet:
                total = sum(trigram_counts[l1][l2].values()) + 13
                for l3 in self.alphabet:
                    self.trigram_probs[l1][l2][l3] = (trigram_counts[l1][l2][l3] + 0.5) / total
    
    def get_letter_probabilities(self, pattern: str, guessed: Set[str]) -> Dict[str, float]:
        """Get probability distribution over letters."""
        length = len(pattern)
        available = set(self.alphabet) - guessed
        
        if not available:
            return {}
        
        scores = defaultdict(float)
        
        # Position-based emissions (HMM)
        if length in self.emission_probs:
            for i, char in enumerate(pattern):
                if char == '_':
                    for letter in available:
                        scores[letter] += self.emission_probs[length][i].get(letter, 0) * 3.0
        
        # Context from revealed letters
        revealed = [c for c in pattern if c != '_']
        
        # Unigram baseline
        for letter in available:
            scores[letter] += self.letter_probs.get(letter, 0) * 1.0
        
        # Bigram context
        if len(revealed) >= 1:
            last = revealed[-1]
            for letter in available:
                scores[letter] += self.bigram_probs[last][letter] * 2.0
        
        # Trigram context
        if len(revealed) >= 2:
            prev2, prev1 = revealed[-2], revealed[-1]
            for letter in available:
                scores[letter] += self.trigram_probs[prev2][prev1][letter] * 2.5
        
        # Normalize to probabilities
        total = sum(scores.values())
        if total > 0:
            return {l: scores[l] / total for l in available}
        return {l: 1.0 / len(available) for l in available}


class PolicyGradientAgent:
    """Policy Gradient agent with HMM-based policy."""
    
    def __init__(self, hmm_model: EnhancedHMMModel,
                 learning_rate: float = 0.001,
                 discount: float = 0.95,
                 epsilon: float = 0.1):
        
        self.alphabet = 'abcdefghijklmnopqrstuvwxyz'
        self.hmm_model = hmm_model
        
        # Policy adjustment weights (start at 1.0 = pure HMM)
        self.policy_weights = defaultdict(lambda: 1.0)
        
        # Hyperparameters
        self.learning_rate = learning_rate
        self.discount = discount
        self.epsilon = epsilon
        
        # Episode buffer
        self.episode_buffer = []
        
        # Statistics
        self.episode_rewards = []
        self.episode_lengths = []
        self.wins = []
    
    def get_action(self, pattern: str, guessed: Set[str], training: bool = True) -> str:
        """
        Get action using HMM policy with learned adjustments and epsilon exploration.
        """
        available = set(self.alphabet) - guessed
        if not available:
            return random.choice(list(self.alphabet))
        
        # Get HMM base probabilities
        hmm_probs = self.hmm_model.get_letter_probabilities(pattern, guessed)
        
        if not hmm_probs:
            return random.choice(list(available))
        
        # Epsilon-greedy exploration during training
        if training and random.random() < self.epsilon:
            # Exploration: weighted random based on HMM
            letters = list(hmm_probs.keys())
            probs = [hmm_probs[l] for l in letters]
            return random.choices(letters, weights=probs)[0]
        
        # Apply learned policy weights
        state_key = self._get_state_key(pattern, guessed)
        adjusted_probs = {}
        
        for letter in available:
            base_prob = hmm_probs.get(letter, 0)
            weight = self.policy_weights.get((state_key, letter), 1.0)
            adjusted_probs[letter] = base_prob * weight
        
        # Normalize
        total = sum(adjusted_probs.values())
        if total > 0:
            adjusted_probs = {l: p / total for l, p in adjusted_probs.items()}
        
        # Select best action
        return max(adjusted_probs.items(), key=lambda x: x[1])[0]
    
    def _get_state_key(self, pattern: str, guessed: Set[str]) -> str:
        """Get simplified state key."""
        num_blanks = pattern.count('_')
        num_guessed = len(guessed)
        return f"{len(pattern)}|{num_blanks}|{num_guessed}"
    
    def store_transition(self, pattern: str, guessed: Set[str], action: str, reward: float):
        """Store transition for policy update."""
        state_key = self._get_state_key(pattern, guessed)
        self.episode_buffer.append((state_key, action, reward))
    
    def update_policy(self, episode_return: float):
        """
        Update policy weights based on episode return.
        Increase weights for actions that led to positive returns.
        """
        if not self.episode_buffer:
            return
        
        # Compute discounted returns
        returns = []
        G = 0
        for _, _, reward in reversed(self.episode_buffer):
            G = reward + self.discount * G
            returns.insert(0, G)
        
        # Normalize returns
        if len(returns) > 1:
            mean_return = np.mean(returns)
            std_return = np.std(returns) + 1e-8
            returns = [(r - mean_return) / std_return for r in returns]
        
        # Update weights
        for (state_key, action, _), G in zip(self.episode_buffer, returns):
            key = (state_key, action)
            current_weight = self.policy_weights[key]
            
            # Policy gradient update: increase weight if G > 0
            update = self.learning_rate * G
            new_weight = current_weight + update
            
            # Clip weights to reasonable range
            self.policy_weights[key] = max(0.1, min(10.0, new_weight))
        
        # Clear buffer
        self.episode_buffer = []


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
            reward = 10 + (num_revealed * 2)
        else:
            self.wrong += 1
            reward = -15
        
        done = self.is_won() or self.is_lost()
        
        if done:
            if self.is_won():
                reward += 100
            else:
                reward -= 50
        
        return self._get_state(), reward, done
    
    def is_won(self) -> bool:
        return '_' not in self.pattern
    
    def is_lost(self) -> bool:
        return self.wrong >= self.max_wrong
    
    def is_done(self) -> bool:
        return self.is_won() or self.is_lost()


def train_policy_gradient(agent: PolicyGradientAgent, words: List[str], 
                         num_episodes: int = 5000):
    """Train policy gradient agent."""
    print(f"\nTraining Policy Gradient Agent for {num_episodes} episodes...")
    
    for episode in range(num_episodes):
        word = random.choice(words)
        game = HangmanGame(word)
        
        state = game.reset()
        total_reward = 0
        steps = 0
        
        while not game.is_done():
            pattern, guessed, wrong = state
            action = agent.get_action(pattern, guessed, training=True)
            
            next_state, reward, done = game.step(action)
            
            # Store for policy update
            agent.store_transition(pattern, guessed, action, reward)
            
            total_reward += reward
            steps += 1
            state = next_state
        
        # Update policy based on episode outcome
        agent.update_policy(total_reward)
        
        agent.episode_rewards.append(total_reward)
        agent.episode_lengths.append(steps)
        agent.wins.append(1 if game.is_won() else 0)
        
        if (episode + 1) % 500 == 0:
            recent_wins = sum(agent.wins[-500:])
            avg_reward = np.mean(agent.episode_rewards[-500:])
            print(f"Episode {episode + 1}/{num_episodes} | "
                  f"Win Rate: {recent_wins/500:.2%} | "
                  f"Avg Reward: {avg_reward:.2f} | "
                  f"Policy Weights: {len(agent.policy_weights)}")
    
    print("Training completed!")
    return agent


def evaluate_agent(agent: PolicyGradientAgent, test_words: List[str]) -> Dict:
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
            action = agent.get_action(pattern, guessed, training=False)
            
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
    print("Hangman Policy Gradient Agent - Version 14")
    print("Actor-Critic RL with HMM-Guided Policy")
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
    
    # Train HMM model
    hmm = EnhancedHMMModel()
    hmm.train(train_words)
    
    # Initialize policy gradient agent
    agent = PolicyGradientAgent(
        hmm_model=hmm,
        learning_rate=0.001,
        discount=0.95,
        epsilon=0.1  # 10% exploration
    )
    
    # Train
    agent = train_policy_gradient(agent, train_words, num_episodes=5000)
    
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
    
    # Save model
    with open('models/policy_gradient_v14.pkl', 'wb') as f:
        pickle.dump({
            'policy_weights': dict(agent.policy_weights),
            'episode_rewards': agent.episode_rewards,
            'wins': agent.wins
        }, f)
    print("\nModel saved to models/policy_gradient_v14.pkl")


if __name__ == "__main__":
    main()
