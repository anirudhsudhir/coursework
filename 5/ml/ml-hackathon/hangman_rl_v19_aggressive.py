"""
Hangman Agent - Aggressive HMM+RL with 90:10 Ratio
Version 19: Minimal RL adjustments on strong HMM

Key Features:
1. 90% HMM baseline, only 10% learned adjustments
2. Enhanced state representation with pattern matching
3. Conservative Q-learning with small updates
4. Longer training (15K episodes)
5. Adaptive learning rate decay
"""

from collections import defaultdict, Counter
from typing import List, Tuple, Set, Dict
import random


class EnhancedHMM:
    """Enhanced HMM with multiple signal sources."""
    
    def __init__(self):
        self.alphabet = 'abcdefghijklmnopqrstuvwxyz'
        self.unigrams = {}
        self.bigrams = defaultdict(lambda: defaultdict(float))
        self.trigrams = defaultdict(lambda: defaultdict(lambda: defaultdict(float)))
        self.quadgrams = defaultdict(lambda: defaultdict(lambda: defaultdict(lambda: defaultdict(float))))
        self.positions = defaultdict(lambda: defaultdict(lambda: defaultdict(float)))
        self.pattern_freq = defaultdict(Counter)
        self.length_letter_freq = defaultdict(Counter)
    
    def train(self, words: List[str]):
        """Train comprehensive HMM."""
        print("Training Enhanced HMM...")
        
        # Unigrams
        letter_counts = Counter()
        for word in words:
            letter_counts.update(word)
        total = sum(letter_counts.values())
        self.unigrams = {l: (letter_counts[l] + 0.5) / (total + 13) for l in self.alphabet}
        
        # Bigrams
        bigram_counts = defaultdict(Counter)
        for word in words:
            for i in range(len(word) - 1):
                bigram_counts[word[i]][word[i+1]] += 1
        
        for l1 in self.alphabet:
            total = sum(bigram_counts[l1].values()) + 13
            for l2 in self.alphabet:
                self.bigrams[l1][l2] = (bigram_counts[l1][l2] + 0.5) / total
        
        # Trigrams
        trigram_counts = defaultdict(lambda: defaultdict(Counter))
        for word in words:
            for i in range(len(word) - 2):
                trigram_counts[word[i]][word[i+1]][word[i+2]] += 1
        
        for l1 in self.alphabet:
            for l2 in self.alphabet:
                total = sum(trigram_counts[l1][l2].values()) + 13
                for l3 in self.alphabet:
                    self.trigrams[l1][l2][l3] = (trigram_counts[l1][l2][l3] + 0.5) / total
        
        # Quadgrams (4-grams)
        quadgram_counts = defaultdict(lambda: defaultdict(lambda: defaultdict(Counter)))
        for word in words:
            for i in range(len(word) - 3):
                quadgram_counts[word[i]][word[i+1]][word[i+2]][word[i+3]] += 1
        
        for l1 in self.alphabet:
            for l2 in self.alphabet:
                for l3 in self.alphabet:
                    total = sum(quadgram_counts[l1][l2][l3].values()) + 13
                    for l4 in self.alphabet:
                        self.quadgrams[l1][l2][l3][l4] = \
                            (quadgram_counts[l1][l2][l3][l4] + 0.5) / total
        
        # Position-based
        position_counts = defaultdict(lambda: defaultdict(Counter))
        for word in words:
            length = len(word)
            for pos, letter in enumerate(word):
                position_counts[length][pos][letter] += 1
                self.length_letter_freq[length][letter] += 1
        
        for length in range(1, 25):
            if length in position_counts:
                for pos in range(length):
                    total = sum(position_counts[length][pos].values()) + 13
                    for letter in self.alphabet:
                        self.positions[length][pos][letter] = \
                            (position_counts[length][pos][letter] + 0.5) / total
        
        # Pattern frequencies
        for word in words:
            # Patterns like consonant-vowel structure
            pattern = ''.join(['V' if c in 'aeiou' else 'C' for c in word])
            for letter in set(word):
                self.pattern_freq[pattern][letter] += 1
    
    def get_probs(self, pattern: str, guessed: Set[str]) -> Dict[str, float]:
        """Get comprehensive probabilities."""
        available = set(self.alphabet) - guessed
        if not available:
            return {}
        
        scores = defaultdict(float)
        length = len(pattern)
        
        # Position-based (strongest signal) - weight 4.0
        if length in self.positions:
            for i, char in enumerate(pattern):
                if char == '_':
                    for letter in available:
                        scores[letter] += self.positions[length][i].get(letter, 0) * 4.0
        
        # Length-specific letter frequencies - weight 1.5
        if length in self.length_letter_freq:
            total = sum(self.length_letter_freq[length].values())
            for letter in available:
                scores[letter] += (self.length_letter_freq[length][letter] / max(total, 1)) * 1.5
        
        # N-gram context
        revealed = [c for c in pattern if c != '_']
        
        # Unigrams - weight 1.0
        for letter in available:
            scores[letter] += self.unigrams.get(letter, 0) * 1.0
        
        # Bigrams - weight 2.5
        if len(revealed) >= 1:
            last = revealed[-1]
            for letter in available:
                scores[letter] += self.bigrams[last][letter] * 2.5
        
        # Trigrams - weight 3.0
        if len(revealed) >= 2:
            prev2, prev1 = revealed[-2], revealed[-1]
            for letter in available:
                scores[letter] += self.trigrams[prev2][prev1][letter] * 3.0
        
        # Quadgrams - weight 3.5
        if len(revealed) >= 3:
            prev3, prev2, prev1 = revealed[-3], revealed[-2], revealed[-1]
            for letter in available:
                scores[letter] += self.quadgrams[prev3][prev2][prev1][letter] * 3.5
        
        # Pattern matching - weight 2.0
        current_pattern = ''.join(['V' if (c in 'aeiou' and c != '_') else ('C' if c != '_' else 'X') 
                                   for c in pattern])
        for letter in available:
            if current_pattern in self.pattern_freq:
                freq = self.pattern_freq[current_pattern][letter]
                total = sum(self.pattern_freq[current_pattern].values())
                if total > 0:
                    scores[letter] += (freq / total) * 2.0
        
        # Normalize
        total = sum(scores.values())
        if total > 0:
            return {l: scores[l] / total for l in available}
        return {l: 1.0 / len(available) for l in available}


class ConservativeRLAgent:
    """RL agent with 90:10 HMM:Learned ratio."""
    
    def __init__(self, hmm: EnhancedHMM,
                 initial_lr: float = 0.002,
                 lr_decay: float = 0.9995,
                 discount: float = 0.95,
                 epsilon: float = 0.02,
                 hmm_weight: float = 0.9):
        
        self.alphabet = 'abcdefghijklmnopqrstuvwxyz'
        self.hmm = hmm
        self.hmm_weight = hmm_weight  # 90% HMM, 10% learned
        
        # Q-learning
        self.q_adjustments = defaultdict(lambda: defaultdict(float))
        self.visit_counts = defaultdict(lambda: defaultdict(int))
        
        self.initial_lr = initial_lr
        self.learning_rate = initial_lr
        self.lr_decay = lr_decay
        self.discount = discount
        self.epsilon = epsilon
        
        # Stats
        self.wins = []
        self.episode_rewards = []
        self.updates = 0
    
    def get_state_key(self, pattern: str, guessed: Set[str], wrong: int) -> str:
        """Create detailed state representation."""
        length = len(pattern)
        blanks = pattern.count('_')
        
        # Pattern structure
        revealed_pos = []
        for i, c in enumerate(pattern):
            if c != '_':
                revealed_pos.append(str(i))
        revealed_str = ','.join(revealed_pos[:5])  # First 5 revealed positions
        
        # Vowel/Consonant pattern
        vc_pattern = ''.join(['V' if (c in 'aeiou' and c != '_') else ('C' if c != '_' else '_') 
                              for c in pattern[:min(8, length)]])
        
        # Last revealed letters
        revealed_letters = ''.join([c for c in pattern if c != '_'][-3:])
        
        return f"{length}_{blanks}_{wrong}_{len(guessed)}_{vc_pattern}_{revealed_letters}"
    
    def choose_action(self, pattern: str, guessed: Set[str], wrong: int, training: bool = True) -> str:
        """Choose action with 90:10 HMM:RL ratio."""
        available = set(self.alphabet) - guessed
        if not available:
            return random.choice(list(self.alphabet))
        
        # Epsilon-greedy exploration
        if training and random.random() < self.epsilon:
            hmm_probs = self.hmm.get_probs(pattern, guessed)
            letters = list(hmm_probs.keys())
            weights = [hmm_probs[l] for l in letters]
            return random.choices(letters, weights=weights)[0]
        
        # Get HMM probabilities
        hmm_probs = self.hmm.get_probs(pattern, guessed)
        
        # Get Q-adjustments
        state_key = self.get_state_key(pattern, guessed, wrong)
        
        # Combine: 90% HMM + 10% learned
        combined_scores = {}
        for letter in available:
            hmm_score = hmm_probs.get(letter, 0)
            q_adj = self.q_adjustments[state_key][letter]
            
            # Small Q-adjustment bounded to prevent destabilizing HMM
            bounded_q = max(-0.05, min(0.05, q_adj))
            
            combined_scores[letter] = self.hmm_weight * hmm_score + (1 - self.hmm_weight) * (hmm_score + bounded_q)
        
        return max(combined_scores.items(), key=lambda x: x[1])[0]
    
    def update(self, state: Tuple, action: str, reward: float, next_state: Tuple, done: bool):
        """Update Q-adjustments."""
        pattern, guessed, wrong = state
        state_key = self.get_state_key(pattern, guessed, wrong)
        
        # Current Q
        current_q = self.q_adjustments[state_key][action]
        
        # Target Q
        if done:
            target = reward
        else:
            next_pattern, next_guessed, next_wrong = next_state
            next_state_key = self.get_state_key(next_pattern, next_guessed, wrong)
            next_available = set(self.alphabet) - next_guessed
            
            if next_available:
                max_next_q = max([self.q_adjustments[next_state_key][a] for a in next_available], default=0)
                target = reward + self.discount * max_next_q
            else:
                target = reward
        
        # Update with decaying learning rate
        self.visit_counts[state_key][action] += 1
        visits = self.visit_counts[state_key][action]
        
        # Adaptive learning rate: decays with visits and episodes
        adaptive_lr = self.learning_rate / (1 + 0.01 * visits)
        
        self.q_adjustments[state_key][action] = current_q + adaptive_lr * (target - current_q)
        self.updates += 1
    
    def decay_learning_rate(self):
        """Decay learning rate."""
        self.learning_rate *= self.lr_decay


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
            num_revealed = sum(1 for i, c in enumerate(self.word) if c == letter and self.pattern[i] == '_')
            for i, c in enumerate(self.word):
                if c == letter:
                    self.pattern[i] = letter
            reward = 15 + (num_revealed * 5)
        else:
            self.wrong += 1
            reward = -25
        
        done = self.is_won() or self.is_lost()
        
        if done:
            if self.is_won():
                reward += 150
            else:
                reward -= 100
        
        return self._get_state(), reward, done
    
    def is_won(self) -> bool:
        return '_' not in self.pattern
    
    def is_lost(self) -> bool:
        return self.wrong >= self.max_wrong
    
    def is_done(self) -> bool:
        return self.is_won() or self.is_lost()


def train_agent(agent: ConservativeRLAgent, words: List[str], num_episodes: int = 15000):
    """Train agent with longer episodes."""
    print(f"\nTraining Conservative RL Agent for {num_episodes} episodes...")
    print(f"HMM Weight: {agent.hmm_weight:.0%}, Learned Weight: {1-agent.hmm_weight:.0%}")
    
    for episode in range(num_episodes):
        word = random.choice(words)
        game = HangmanGame(word)
        
        state = game.reset()
        total_reward = 0
        
        while not game.is_done():
            pattern, guessed, wrong = state
            action = agent.choose_action(pattern, guessed, wrong, training=True)
            
            next_state, reward, done = game.step(action)
            
            agent.update(state, action, reward, next_state, done)
            
            total_reward += reward
            state = next_state
        
        agent.decay_learning_rate()
        agent.wins.append(1 if game.is_won() else 0)
        agent.episode_rewards.append(total_reward)
        
        if (episode + 1) % 500 == 0:
            recent_wins = sum(agent.wins[-500:])
            avg_reward = sum(agent.episode_rewards[-500:]) / 500
            print(f"Episode {episode + 1}/{num_episodes} | "
                  f"Win Rate: {recent_wins/500:.2%} | "
                  f"Avg Reward: {avg_reward:.2f} | "
                  f"LR: {agent.learning_rate:.6f} | "
                  f"Updates: {agent.updates} | "
                  f"States: {len(agent.q_adjustments)}")
    
    print("Training completed!")
    return agent


def evaluate_agent(agent: ConservativeRLAgent, test_words: List[str]) -> Dict:
    """Evaluate agent."""
    print(f"\nEvaluating on {len(test_words)} test words...")
    
    wins = 0
    total_wrong = 0
    total_repeated = 0
    
    for i, word in enumerate(test_words):
        game = HangmanGame(word)
        state = game.reset()
        
        while not game.is_done():
            pattern, guessed, wrong = state
            action = agent.choose_action(pattern, guessed, wrong, training=False)
            
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
    print("Hangman Conservative RL Agent - Version 19")
    print("90% HMM Baseline + 10% Learned Adjustments")
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
    
    # Train HMM
    hmm = EnhancedHMM()
    hmm.train(train_words)
    
    # Initialize agent
    agent = ConservativeRLAgent(
        hmm=hmm,
        initial_lr=0.002,
        lr_decay=0.9995,
        discount=0.95,
        epsilon=0.02,
        hmm_weight=0.9  # 90% HMM, 10% learned
    )
    
    # Train
    agent = train_agent(agent, train_words, num_episodes=15000)
    
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
    print(f"\nLearning Statistics:")
    print(f"Total Q-updates: {agent.updates}")
    print(f"Unique States: {len(agent.q_adjustments)}")
    print(f"Final Learning Rate: {agent.learning_rate:.6f}")
    print("=" * 70)


if __name__ == "__main__":
    main()
