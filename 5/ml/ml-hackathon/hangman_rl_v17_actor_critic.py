"""
Hangman Agent - Actor-Critic with Advantage Learning
Version 17: Dual network with HMM baseline

Key Features:
1. Actor network: learns policy improvements over HMM
2. Critic network: estimates state values
3. Advantage function: A(s,a) = Q(s,a) - V(s)
4. HMM provides strong baseline, actor learns refinements
5. Continuous learning with small adjustments
"""

from collections import defaultdict, Counter, deque
from typing import List, Tuple, Set, Dict
import random
import math


class HMMBaseline:
    """HMM baseline for actor-critic."""
    
    def __init__(self):
        self.alphabet = 'abcdefghijklmnopqrstuvwxyz'
        self.unigrams = {}
        self.bigrams = defaultdict(lambda: defaultdict(float))
        self.trigrams = defaultdict(lambda: defaultdict(lambda: defaultdict(float)))
        self.positions = defaultdict(lambda: defaultdict(lambda: defaultdict(float)))
        self.length_patterns = defaultdict(Counter)
    
    def train(self, words: List[str]):
        """Train HMM baseline."""
        print("Training HMM baseline...")
        
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
                        self.positions[length][pos][letter] = \
                            (position_counts[length][pos][letter] + 0.5) / total
        
        # Length patterns
        for word in words:
            self.length_patterns[len(word)].update(word)
    
    def get_baseline_probs(self, pattern: str, guessed: Set[str]) -> Dict[str, float]:
        """Get baseline probabilities."""
        available = set(self.alphabet) - guessed
        scores = defaultdict(float)
        length = len(pattern)
        
        # Position-based (strongest signal)
        if length in self.positions:
            for i, char in enumerate(pattern):
                if char == '_':
                    for letter in available:
                        scores[letter] += self.positions[length][i].get(letter, 0) * 3.5
        
        # Context from revealed letters
        revealed = [c for c in pattern if c != '_']
        
        for letter in available:
            scores[letter] += self.unigrams.get(letter, 0) * 1.0
        
        if len(revealed) >= 1:
            last = revealed[-1]
            for letter in available:
                scores[letter] += self.bigrams[last][letter] * 2.5
        
        if len(revealed) >= 2:
            prev2, prev1 = revealed[-2], revealed[-1]
            for letter in available:
                scores[letter] += self.trigrams[prev2][prev1][letter] * 3.0
        
        # Length-specific frequencies
        if length in self.length_patterns:
            total = sum(self.length_patterns[length].values())
            for letter in available:
                scores[letter] += (self.length_patterns[length][letter] / total) * 1.5
        
        # Normalize
        total = sum(scores.values())
        if total > 0:
            return {l: scores[l] / total for l in available}
        return {l: 1.0 / len(available) for l in available}


class ActorCriticNetwork:
    """Actor-Critic with advantage learning."""
    
    def __init__(self, state_dim: int, action_dim: int = 26):
        # Actor: policy improvements (additive to baseline)
        self.actor_weights = defaultdict(lambda: defaultdict(float))
        
        # Critic: state value estimates
        self.critic_weights = defaultdict(float)
        
        self.state_dim = state_dim
        self.action_dim = action_dim
        
        # Statistics
        self.actor_updates = 0
        self.critic_updates = 0
    
    def get_state_key(self, pattern: str, guessed: Set[str], wrong: int) -> str:
        """Create state hash."""
        blanks = pattern.count('_')
        length = len(pattern)
        revealed_positions = ''.join(['1' if c != '_' else '0' for c in pattern[:min(5, length)]])
        return f"{length}_{blanks}_{wrong}_{len(guessed)}_{revealed_positions}"
    
    def get_actor_adjustment(self, state_key: str, action: str) -> float:
        """Get actor's policy adjustment for action."""
        return self.actor_weights[state_key][action]
    
    def get_state_value(self, state_key: str) -> float:
        """Get critic's state value estimate."""
        return self.critic_weights[state_key]
    
    def update_critic(self, state_key: str, td_error: float, learning_rate: float = 0.01):
        """Update critic using TD error."""
        self.critic_weights[state_key] += learning_rate * td_error
        self.critic_updates += 1
    
    def update_actor(self, state_key: str, action: str, advantage: float, learning_rate: float = 0.005):
        """Update actor using advantage."""
        # Update policy weights in direction of advantage
        self.actor_weights[state_key][action] += learning_rate * advantage
        self.actor_updates += 1


class ActorCriticAgent:
    """Actor-Critic agent with HMM baseline."""
    
    def __init__(self, hmm_baseline: HMMBaseline,
                 actor_lr: float = 0.005,
                 critic_lr: float = 0.01,
                 discount: float = 0.95,
                 epsilon: float = 0.03,
                 baseline_weight: float = 0.85):
        
        self.alphabet = 'abcdefghijklmnopqrstuvwxyz'
        self.hmm = hmm_baseline
        self.network = ActorCriticNetwork(state_dim=20)
        
        self.actor_lr = actor_lr
        self.critic_lr = critic_lr
        self.discount = discount
        self.epsilon = epsilon
        self.baseline_weight = baseline_weight  # How much to trust HMM vs learned
        
        # Training stats
        self.episode_rewards = []
        self.wins = []
        self.episode_lengths = []
    
    def choose_action(self, pattern: str, guessed: Set[str], wrong: int, training: bool = True) -> str:
        """Choose action using actor-critic policy."""
        available = set(self.alphabet) - guessed
        if not available:
            return random.choice(list(self.alphabet))
        
        # Epsilon-greedy exploration
        if training and random.random() < self.epsilon:
            hmm_probs = self.hmm.get_baseline_probs(pattern, guessed)
            letters = list(hmm_probs.keys())
            weights = [hmm_probs[l] for l in letters]
            return random.choices(letters, weights=weights)[0]
        
        # Get baseline probabilities
        hmm_probs = self.hmm.get_baseline_probs(pattern, guessed)
        
        # Get actor adjustments
        state_key = self.network.get_state_key(pattern, guessed, wrong)
        
        combined_scores = {}
        for letter in available:
            baseline_score = hmm_probs.get(letter, 0)
            actor_adjustment = self.network.get_actor_adjustment(state_key, letter)
            
            # Combine: mostly baseline, small learned adjustments
            combined_scores[letter] = (self.baseline_weight * baseline_score + 
                                      (1 - self.baseline_weight) * math.tanh(actor_adjustment))
        
        # Normalize
        total = sum(combined_scores.values())
        if total > 0:
            combined_scores = {l: s/total for l, s in combined_scores.items()}
        
        return max(combined_scores.items(), key=lambda x: x[1])[0]
    
    def update(self, state: Tuple, action: str, reward: float, next_state: Tuple, done: bool):
        """Update actor and critic."""
        pattern, guessed, wrong = state
        next_pattern, next_guessed, next_wrong = next_state
        
        state_key = self.network.get_state_key(pattern, guessed, wrong)
        next_state_key = self.network.get_state_key(next_pattern, next_guessed, next_wrong)
        
        # Get current and next state values
        v_current = self.network.get_state_value(state_key)
        v_next = 0 if done else self.network.get_state_value(next_state_key)
        
        # TD error
        td_target = reward + self.discount * v_next
        td_error = td_target - v_current
        
        # Update critic
        self.network.update_critic(state_key, td_error, self.critic_lr)
        
        # Advantage = TD error (since we're using state-value baseline)
        advantage = td_error
        
        # Update actor (only if significant advantage)
        if abs(advantage) > 0.1:  # Threshold to prevent noise
            self.network.update_actor(state_key, action, advantage, self.actor_lr)


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
            reward = 12 + (num_revealed * 4)
        else:
            self.wrong += 1
            reward = -20
        
        done = self.is_won() or self.is_lost()
        
        if done:
            if self.is_won():
                reward += 120
            else:
                reward -= 100
        
        return self._get_state(), reward, done
    
    def is_won(self) -> bool:
        return '_' not in self.pattern
    
    def is_lost(self) -> bool:
        return self.wrong >= self.max_wrong
    
    def is_done(self) -> bool:
        return self.is_won() or self.is_lost()


def train_agent(agent: ActorCriticAgent, words: List[str], num_episodes: int = 10000):
    """Train actor-critic agent."""
    print(f"\nTraining Actor-Critic Agent for {num_episodes} episodes...")
    print(f"Baseline weight: {agent.baseline_weight:.0%}, Learned weight: {1-agent.baseline_weight:.0%}")
    
    for episode in range(num_episodes):
        word = random.choice(words)
        game = HangmanGame(word)
        
        state = game.reset()
        total_reward = 0
        steps = 0
        
        while not game.is_done():
            pattern, guessed, wrong = state
            action = agent.choose_action(pattern, guessed, wrong, training=True)
            
            next_state, reward, done = game.step(action)
            
            # Update networks
            agent.update(state, action, reward, next_state, done)
            
            total_reward += reward
            steps += 1
            state = next_state
        
        agent.episode_rewards.append(total_reward)
        agent.wins.append(1 if game.is_won() else 0)
        agent.episode_lengths.append(steps)
        
        if (episode + 1) % 500 == 0:
            recent_wins = sum(agent.wins[-500:])
            avg_reward = sum(agent.episode_rewards[-500:]) / 500
            avg_length = sum(agent.episode_lengths[-500:]) / 500
            print(f"Episode {episode + 1}/{num_episodes} | "
                  f"Win Rate: {recent_wins/500:.2%} | "
                  f"Avg Reward: {avg_reward:.2f} | "
                  f"Avg Steps: {avg_length:.1f} | "
                  f"Actor Updates: {agent.network.actor_updates} | "
                  f"Critic Updates: {agent.network.critic_updates}")
    
    print("Training completed!")
    return agent


def evaluate_agent(agent: ActorCriticAgent, test_words: List[str]) -> Dict:
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
    print("Hangman Actor-Critic Agent - Version 17")
    print("Dual Network with Advantage Learning")
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
    
    # Train HMM baseline
    hmm = HMMBaseline()
    hmm.train(train_words)
    
    # Initialize agent
    agent = ActorCriticAgent(
        hmm_baseline=hmm,
        actor_lr=0.005,
        critic_lr=0.01,
        discount=0.95,
        epsilon=0.03,
        baseline_weight=0.85  # 85% baseline, 15% learned
    )
    
    # Train
    agent = train_agent(agent, train_words, num_episodes=10000)
    
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
    print(f"\nNetwork Statistics:")
    print(f"Actor Updates: {agent.network.actor_updates}")
    print(f"Critic Updates: {agent.network.critic_updates}")
    print(f"Unique States Visited: {len(agent.network.critic_weights)}")
    print("=" * 70)


if __name__ == "__main__":
    main()
