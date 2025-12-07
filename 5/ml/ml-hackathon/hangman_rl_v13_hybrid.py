"""
Hangman Agent - Hybrid HMM-Initialized Q-Learning
Version 13: True RL with HMM-Warm-Started Q-Values

Key Features:
1. Initialize Q-values using HMM probabilities (warm start)
2. Q-Learning with epsilon-greedy exploration
3. Combine learned Q-values with HMM priors during action selection
4. Proper exploration-exploitation trade-off
5. Target: 30%+ success rate with TRUE RL
"""

import numpy as np
from collections import defaultdict, Counter
from typing import List, Tuple, Set, Dict
import random
import pickle


class HMMPriorModel:
    """Lightweight HMM for initializing Q-values and providing priors."""
    
    def __init__(self):
        self.alphabet = 'abcdefghijklmnopqrstuvwxyz'
        self.letter_probs = defaultdict(float)
        self.bigram_probs = defaultdict(lambda: defaultdict(float))
        self.trigram_probs = defaultdict(lambda: defaultdict(lambda: defaultdict(float)))
        self.position_probs = defaultdict(lambda: defaultdict(lambda: defaultdict(float)))
    
    def train(self, words: List[str]):
        """Train HMM priors from corpus."""
        print("Training HMM priors for Q-value initialization...")
        
        # Unigrams
        letter_counts = Counter()
        for word in words:
            letter_counts.update(word)
        total = sum(letter_counts.values())
        for letter in self.alphabet:
            self.letter_probs[letter] = (letter_counts[letter] + 0.1) / (total + 2.6)
        
        # Bigrams
        bigram_counts = defaultdict(Counter)
        for word in words:
            for i in range(len(word) - 1):
                bigram_counts[word[i]][word[i+1]] += 1
        
        for letter1 in self.alphabet:
            total = sum(bigram_counts[letter1].values()) + 2.6
            for letter2 in self.alphabet:
                self.bigram_probs[letter1][letter2] = (bigram_counts[letter1][letter2] + 0.1) / total
        
        # Trigrams
        trigram_counts = defaultdict(lambda: defaultdict(Counter))
        for word in words:
            for i in range(len(word) - 2):
                trigram_counts[word[i]][word[i+1]][word[i+2]] += 1
        
        for l1 in self.alphabet:
            for l2 in self.alphabet:
                total = sum(trigram_counts[l1][l2].values()) + 2.6
                for l3 in self.alphabet:
                    self.trigram_probs[l1][l2][l3] = (trigram_counts[l1][l2][l3] + 0.1) / total
        
        # Position-based (by word length)
        position_counts = defaultdict(lambda: defaultdict(Counter))
        for word in words:
            length = len(word)
            for pos, letter in enumerate(word):
                position_counts[length][pos][letter] += 1
        
        for length in range(1, 25):
            if length in position_counts:
                for pos in range(length):
                    total = sum(position_counts[length][pos].values()) + 2.6
                    for letter in self.alphabet:
                        self.position_probs[length][pos][letter] = \
                            (position_counts[length][pos][letter] + 0.1) / total
    
    def get_prior_score(self, pattern: str, guessed: Set[str], letter: str) -> float:
        """Get HMM prior score for a letter."""
        if letter in guessed:
            return 0.0
        
        score = self.letter_probs[letter] * 2.0  # Base frequency
        
        # Add context from revealed letters
        revealed = [c for c in pattern if c != '_']
        if len(revealed) >= 1:
            last = revealed[-1]
            score += self.bigram_probs[last][letter] * 3.0
        
        if len(revealed) >= 2:
            prev2, prev1 = revealed[-2], revealed[-1]
            score += self.trigram_probs[prev2][prev1][letter] * 4.0
        
        # Add position information
        length = len(pattern)
        if length in self.position_probs:
            blank_positions = [i for i, c in enumerate(pattern) if c == '_']
            for pos in blank_positions[:3]:  # Consider first 3 blanks
                score += self.position_probs[length][pos][letter] * 2.0
        
        return score


class HybridQLearningAgent:
    """Q-Learning agent with HMM-initialized Q-values."""
    
    def __init__(self, hmm_prior: HMMPriorModel = None,
                 learning_rate: float = 0.03,
                 discount: float = 0.95,
                 epsilon_start: float = 0.2,
                 epsilon_end: float = 0.02,
                 epsilon_decay: float = 0.9995):
        
        self.alphabet = 'abcdefghijklmnopqrstuvwxyz'
        self.letter_to_idx = {letter: i for i, letter in enumerate(self.alphabet)}
        
        self.hmm_prior = hmm_prior
        
        # Q-table: (state_hash, action) -> Q-value
        self.q_table = defaultdict(float)
        
        # Track which states have been initialized
        self.initialized_states = set()
        
        # Hyperparameters
        self.learning_rate = learning_rate
        self.discount = discount
        self.epsilon = epsilon_start
        self.epsilon_end = epsilon_end
        self.epsilon_decay = epsilon_decay
        
        # Statistics
        self.episode_rewards = []
        self.episode_lengths = []
        self.wins = []
    
    def _hash_state(self, pattern: str, guessed: Set[str], wrong: int) -> str:
        """Create compact state hash."""
        # Pattern encoding: preserve structure
        pattern_encoded = pattern.replace('_', '.')
        
        # Sort guessed for consistency
        guessed_sorted = ''.join(sorted(guessed))
        
        # Hash: length|pattern_structure|num_guessed|wrong
        return f"{len(pattern)}|{pattern_encoded[:5]}|{len(guessed)}|{wrong}"
    
    def _initialize_q_values(self, state_hash: str, pattern: str, guessed: Set[str]):
        """Initialize Q-values for a new state using HMM priors."""
        if state_hash in self.initialized_states or not self.hmm_prior:
            return
        
        available = set(self.alphabet) - guessed
        
        # Get HMM scores for all available letters
        hmm_scores = {}
        for letter in available:
            hmm_scores[letter] = self.hmm_prior.get_prior_score(pattern, guessed, letter)
        
        # Normalize scores to [-1, 1] range and use as initial Q-values
        if hmm_scores:
            max_score = max(hmm_scores.values())
            min_score = min(hmm_scores.values())
            score_range = max_score - min_score if max_score > min_score else 1.0
            
            for letter in available:
                normalized = ((hmm_scores[letter] - min_score) / score_range) * 2 - 1
                self.q_table[(state_hash, letter)] = normalized * 10  # Scale to meaningful range
        
        self.initialized_states.add(state_hash)
    
    def get_q_value(self, state_hash: str, action: str) -> float:
        """Get Q(s, a)."""
        return self.q_table.get((state_hash, action), 0.0)
    
    def get_action(self, pattern: str, guessed: Set[str], wrong: int, training: bool = True) -> str:
        """Choose action using epsilon-greedy with HMM fallback."""
        available = set(self.alphabet) - guessed
        if not available:
            return random.choice(list(self.alphabet))
        
        state_hash = self._hash_state(pattern, guessed, wrong)
        
        # Initialize Q-values if first visit
        self._initialize_q_values(state_hash, pattern, guessed)
        
        # Epsilon-greedy
        if training and random.random() < self.epsilon:
            # Exploration: use HMM-weighted random selection
            if self.hmm_prior:
                hmm_scores = {l: self.hmm_prior.get_prior_score(pattern, guessed, l) 
                             for l in available}
                total = sum(hmm_scores.values())
                if total > 0:
                    probs = [hmm_scores[l] / total for l in available]
                    return random.choices(list(available), weights=probs)[0]
            return random.choice(list(available))
        
        # Exploitation: combine Q-values with HMM priors
        best_action = None
        best_score = float('-inf')
        
        for action in available:
            q_value = self.get_q_value(state_hash, action)
            
            # Combine Q-value with HMM prior (80% learned, 20% prior)
            if self.hmm_prior:
                hmm_score = self.hmm_prior.get_prior_score(pattern, guessed, action)
                combined_score = 0.8 * q_value + 0.2 * hmm_score
            else:
                combined_score = q_value
            
            if combined_score > best_score:
                best_score = combined_score
                best_action = action
        
        return best_action if best_action else random.choice(list(available))
    
    def update(self, state: Tuple, action: str, reward: float, 
               next_state: Tuple, done: bool):
        """Q-Learning update."""
        pattern, guessed, wrong = state
        next_pattern, next_guessed, next_wrong = next_state
        
        state_hash = self._hash_state(pattern, guessed, wrong)
        next_state_hash = self._hash_state(next_pattern, next_guessed, next_wrong)
        
        # Initialize if needed
        self._initialize_q_values(state_hash, pattern, guessed)
        self._initialize_q_values(next_state_hash, next_pattern, next_guessed)
        
        current_q = self.get_q_value(state_hash, action)
        
        if done:
            target = reward
        else:
            # Max Q-value over next available actions
            next_available = set(self.alphabet) - next_guessed
            max_next_q = max([self.get_q_value(next_state_hash, a) for a in next_available], 
                            default=0.0)
            target = reward + self.discount * max_next_q
        
        # Q-learning update
        td_error = target - current_q
        new_q = current_q + self.learning_rate * td_error
        self.q_table[(state_hash, action)] = new_q
    
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
            # Correct guess
            num_revealed = 0
            for i, char in enumerate(self.word):
                if char == letter:
                    self.pattern[i] = letter
                    num_revealed += 1
            reward = 10 + (num_revealed * 2)  # Bonus for multiple occurrences
        else:
            # Wrong guess
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


def train_hybrid_agent(agent: HybridQLearningAgent, words: List[str], 
                       num_episodes: int = 10000):
    """Train hybrid Q-learning agent."""
    print(f"\nTraining Hybrid Q-Learning Agent for {num_episodes} episodes...")
    
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
            agent.update(state, action, reward, next_state, done)
            
            total_reward += reward
            steps += 1
            state = next_state
        
        agent.decay_epsilon()
        agent.episode_rewards.append(total_reward)
        agent.episode_lengths.append(steps)
        agent.wins.append(1 if game.is_won() else 0)
        
        if (episode + 1) % 500 == 0:
            recent_wins = sum(agent.wins[-500:])
            avg_reward = np.mean(agent.episode_rewards[-500:])
            print(f"Episode {episode + 1}/{num_episodes} | "
                  f"Win Rate: {recent_wins/500:.2%} | "
                  f"Avg Reward: {avg_reward:.2f} | "
                  f"Epsilon: {agent.epsilon:.4f} | "
                  f"Q-table: {len(agent.q_table)}")
    
    print("Training completed!")
    return agent


def evaluate_agent(agent: HybridQLearningAgent, test_words: List[str]) -> Dict:
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
    print("Hangman Hybrid Q-Learning Agent - Version 13")
    print("True RL with HMM-Initialized Q-Values")
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
    
    # Initialize agent with HMM priors
    agent = HybridQLearningAgent(
        hmm_prior=hmm,
        learning_rate=0.03,
        discount=0.95,
        epsilon_start=0.2,
        epsilon_end=0.02,
        epsilon_decay=0.9995
    )
    
    # Train
    agent = train_hybrid_agent(agent, train_words, num_episodes=10000)
    
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
    with open('models/hybrid_qlearning_v13.pkl', 'wb') as f:
        pickle.dump({
            'q_table': dict(agent.q_table),
            'episode_rewards': agent.episode_rewards,
            'wins': agent.wins
        }, f)
    print("\nModel saved to models/hybrid_qlearning_v13.pkl")


if __name__ == "__main__":
    main()
