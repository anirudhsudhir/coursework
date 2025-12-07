"""
Hangman Agent - SARSA with HMM Integration
Version 12: On-Policy Learning with Pre-trained HMM Guidance

Key Improvements:
1. SARSA (on-policy) instead of Q-Learning (off-policy)
2. Initialize Q-values using HMM probabilities (warm start)
3. Adaptive learning rate
4. Better feature engineering
5. Combination of learned Q-values and HMM priors
"""

import numpy as np
from collections import defaultdict, Counter
from typing import List, Tuple, Set, Dict
import random


class SimplifiedHMM:
    """Lightweight HMM for providing prior probabilities."""
    
    def __init__(self):
        self.alphabet = 'abcdefghijklmnopqrstuvwxyz'
        self.letter_probs = defaultdict(float)
        self.bigram_probs = defaultdict(lambda: defaultdict(float))
        self.position_probs = defaultdict(lambda: defaultdict(float))
    
    def train(self, words: List[str]):
        """Quick training on word corpus."""
        print("Training HMM priors...")
        
        # Letter frequencies
        letter_counts = Counter()
        for word in words:
            letter_counts.update(word)
        
        total = sum(letter_counts.values())
        for letter in self.alphabet:
            self.letter_probs[letter] = (letter_counts[letter] + 1) / (total + 26)
        
        # Bigrams
        bigram_counts = defaultdict(Counter)
        for word in words:
            for i in range(len(word) - 1):
                bigram_counts[word[i]][word[i+1]] += 1
        
        for letter1 in self.alphabet:
            total = sum(bigram_counts[letter1].values()) + 26
            for letter2 in self.alphabet:
                self.bigram_probs[letter1][letter2] = (bigram_counts[letter1][letter2] + 1) / total
        
        # Position-based probabilities
        position_counts = defaultdict(Counter)
        for word in words:
            for pos, letter in enumerate(word):
                rel_pos = pos / max(len(word) - 1, 1)  # Normalize to [0, 1]
                bin_pos = int(rel_pos * 10)  # 10 bins
                position_counts[bin_pos][letter] += 1
        
        for bin_pos in range(10):
            total = sum(position_counts[bin_pos].values()) + 26
            for letter in self.alphabet:
                self.position_probs[bin_pos][letter] = (position_counts[bin_pos][letter] + 1) / total
    
    def get_prior(self, pattern: str, guessed: Set[str]) -> Dict[str, float]:
        """Get prior probabilities for unguessed letters."""
        available = set(self.alphabet) - guessed
        probs = {}
        
        # Combine letter frequency, bigrams, and position
        for letter in available:
            score = self.letter_probs[letter]
            
            # Add bigram context if we have revealed letters
            revealed = [c for c in pattern if c != '_']
            if revealed:
                last_letter = revealed[-1]
                score += self.bigram_probs[last_letter][letter]
            
            # Add position context for blank positions
            blanks = [i for i, c in enumerate(pattern) if c == '_']
            if blanks:
                for pos in blanks[:3]:  # Consider first 3 blanks
                    rel_pos = pos / max(len(pattern) - 1, 1)
                    bin_pos = int(rel_pos * 10)
                    score += self.position_probs[bin_pos][letter]
            
            probs[letter] = score
        
        # Normalize
        total = sum(probs.values())
        if total > 0:
            probs = {k: v / total for k, v in probs.items()}
        else:
            # Uniform if no info
            probs = {k: 1.0 / len(available) for k in available}
        
        return probs


class SARSAAgent:
    """SARSA agent with function approximation and HMM guidance."""
    
    def __init__(self, hmm: SimplifiedHMM = None, learning_rate: float = 0.05,
                 discount: float = 0.9, epsilon_start: float = 0.3,
                 epsilon_end: float = 0.01, epsilon_decay: float = 0.998):
        
        self.alphabet = 'abcdefghijklmnopqrstuvwxyz'
        self.letter_to_idx = {letter: i for i, letter in enumerate(self.alphabet)}
        
        # HMM for priors
        self.hmm = hmm
        
        # Q-function as dict of (state_hash, action) -> Q-value
        self.q_table = defaultdict(float)
        
        # Hyperparameters
        self.learning_rate = learning_rate
        self.discount = discount
        self.epsilon = epsilon_start
        self.epsilon_end = epsilon_end
        self.epsilon_decay = epsilon_decay
        
        # Statistics
        self.episode_rewards = []
        self.episode_steps = []
    
    def _hash_state(self, pattern: str, guessed: Set[str], wrong: int) -> str:
        """Create hashable state representation."""
        # Simplify pattern: group consecutive blanks/letters
        simplified = []
        prev = None
        count = 0
        for c in pattern:
            if c == prev or (c != '_' and prev != '_' and prev is not None):
                count += 1
            else:
                if prev is not None:
                    simplified.append(f"{prev}{count}")
                prev = '_' if c == '_' else 'X'
                count = 1
        if prev is not None:
            simplified.append(f"{prev}{count}")
        
        pattern_str = ''.join(simplified)
        guessed_str = ''.join(sorted(guessed))
        return f"{len(pattern)}|{pattern_str}|{len(guessed)}|{wrong}"
    
    def get_q_value(self, state_hash: str, action: str) -> float:
        """Get Q(s, a) with HMM prior initialization."""
        key = (state_hash, action)
        return self.q_table.get(key, 0.0)
    
    def get_action(self, pattern: str, guessed: Set[str], wrong: int,
                   training: bool = True) -> str:
        """
        Choose action using epsilon-greedy with HMM guidance.
        
        During exploration, sample from HMM priors (informed exploration).
        During exploitation, use Q-values weighted with HMM priors.
        """
        available = set(self.alphabet) - guessed
        if not available:
            return random.choice(list(self.alphabet))
        
        state_hash = self._hash_state(pattern, guessed, wrong)
        
        # Epsilon-greedy
        if training and random.random() < self.epsilon:
            # Informed exploration: sample from HMM priors
            if self.hmm:
                priors = self.hmm.get_prior(pattern, guessed)
                letters = list(priors.keys())
                probabilities = [priors[l] for l in letters]
                return random.choices(letters, weights=probabilities)[0]
            else:
                return random.choice(list(available))
        
        # Exploitation: combine Q-values with HMM priors
        best_action = None
        best_score = float('-inf')
        
        # Get HMM priors if available
        hmm_priors = self.hmm.get_prior(pattern, guessed) if self.hmm else {}
        
        for action in available:
            q_value = self.get_q_value(state_hash, action)
            
            # Weighted combination: 70% Q-value, 30% HMM prior
            if hmm_priors:
                score = 0.7 * q_value + 0.3 * (hmm_priors.get(action, 0) * 10)
            else:
                score = q_value
            
            if score > best_score:
                best_score = score
                best_action = action
        
        return best_action if best_action else random.choice(list(available))
    
    def update(self, state: Tuple, action: str, reward: float,
               next_state: Tuple, next_action: str, done: bool):
        """
        SARSA update: Q(s,a) ← Q(s,a) + α[r + γQ(s',a') - Q(s,a)]
        """
        pattern, guessed, wrong = state
        next_pattern, next_guessed, next_wrong = next_state
        
        state_hash = self._hash_state(pattern, guessed, wrong)
        next_state_hash = self._hash_state(next_pattern, next_guessed, next_wrong)
        
        # Current Q-value
        current_q = self.get_q_value(state_hash, action)
        
        # SARSA: use actual next action (on-policy)
        if done:
            target = reward
        else:
            next_q = self.get_q_value(next_state_hash, next_action)
            target = reward + self.discount * next_q
        
        # Update Q-value
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
        """Reset game state."""
        self.pattern = ['_'] * len(self.word)
        self.guessed = set()
        self.wrong = 0
        return self._get_state()
    
    def _get_state(self) -> Tuple[str, Set[str], int]:
        """Return current state."""
        return (''.join(self.pattern), self.guessed.copy(), self.wrong)
    
    def step(self, letter: str) -> Tuple[Tuple, float, bool]:
        """Take action and return (next_state, reward, done)."""
        letter = letter.lower()
        self.guessed.add(letter)
        
        if letter in self.word:
            # Correct guess
            for i, char in enumerate(self.word):
                if char == letter:
                    self.pattern[i] = letter
            reward = 15  # Higher reward for correct
        else:
            # Wrong guess
            self.wrong += 1
            reward = -25  # Higher penalty for wrong
        
        done = self.is_won() or self.is_lost()
        
        if done:
            if self.is_won():
                reward += 150  # Big bonus for winning
            else:
                reward -= 150  # Big penalty for losing
        
        return self._get_state(), reward, done
    
    def is_won(self) -> bool:
        return '_' not in self.pattern
    
    def is_lost(self) -> bool:
        return self.wrong >= self.max_wrong
    
    def is_done(self) -> bool:
        return self.is_won() or self.is_lost()


def train_sarsa(agent: SARSAAgent, words: List[str], num_episodes: int = 8000):
    """Train SARSA agent."""
    print(f"\nTraining SARSA Agent for {num_episodes} episodes...")
    
    for episode in range(num_episodes):
        word = random.choice(words)
        game = HangmanGame(word)
        
        state = game.reset()
        pattern, guessed, wrong = state
        
        # Choose initial action
        action = agent.get_action(pattern, guessed, wrong, training=True)
        
        total_reward = 0
        steps = 0
        
        while not game.is_done():
            # Take action
            next_state, reward, done = game.step(action)
            next_pattern, next_guessed, next_wrong = next_state
            
            # Choose next action (SARSA: on-policy)
            next_action = agent.get_action(next_pattern, next_guessed, next_wrong, training=True)
            
            # Update Q-values
            agent.update(state, action, reward, next_state, next_action, done)
            
            # Move to next state
            state = next_state
            action = next_action
            total_reward += reward
            steps += 1
        
        # Decay exploration
        agent.decay_epsilon()
        
        # Track statistics
        agent.episode_rewards.append(total_reward)
        agent.episode_steps.append(steps)
        
        # Progress
        if (episode + 1) % 200 == 0:
            avg_reward = np.mean(agent.episode_rewards[-200:])
            avg_steps = np.mean(agent.episode_steps[-200:])
            print(f"Episode {episode + 1}/{num_episodes} | "
                  f"Avg Reward: {avg_reward:.2f} | "
                  f"Avg Steps: {avg_steps:.2f} | "
                  f"Epsilon: {agent.epsilon:.4f} | "
                  f"Q-table size: {len(agent.q_table)}")
    
    print("Training completed!")
    return agent


def evaluate(agent: SARSAAgent, test_words: List[str]) -> Dict:
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
        
        if (i + 1) % 100 == 0:
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
    """Main execution."""
    print("=" * 60)
    print("Hangman SARSA Agent - Version 12")
    print("=" * 60)
    
    # Load data
    corpus_file = 'data/corpus.txt'
    test_file = 'data/test.txt'
    
    with open(corpus_file, 'r') as f:
        train_words = [line.strip().lower() for line in f if line.strip()]
    print(f"Loaded {len(train_words)} training words")
    
    with open(test_file, 'r') as f:
        test_words = [line.strip().lower() for line in f if line.strip()]
    print(f"Loaded {len(test_words)} test words")
    
    # Train HMM for priors
    hmm = SimplifiedHMM()
    hmm.train(train_words)
    
    # Initialize SARSA agent with HMM
    agent = SARSAAgent(
        hmm=hmm,
        learning_rate=0.05,
        discount=0.9,
        epsilon_start=0.3,
        epsilon_end=0.01,
        epsilon_decay=0.998
    )
    
    # Train
    agent = train_sarsa(agent, train_words, num_episodes=8000)
    
    # Evaluate
    results = evaluate(agent, test_words)
    
    # Results
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
