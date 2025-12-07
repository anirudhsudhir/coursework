"""
Hangman Agent - Monte Carlo Tree Search with UCB
Version 18: MCTS with HMM-guided rollouts

Key Features:
1. MCTS for lookahead planning
2. UCB1 for exploration-exploitation balance
3. HMM-guided simulations for faster convergence
4. Limited search budget for practical runtime
5. Value backpropagation through tree
"""

from collections import defaultdict, Counter
from typing import List, Tuple, Set, Dict, Optional
import random
import math


class HMMGuide:
    """HMM for guiding MCTS simulations."""
    
    def __init__(self):
        self.alphabet = 'abcdefghijklmnopqrstuvwxyz'
        self.unigrams = {}
        self.bigrams = defaultdict(lambda: defaultdict(float))
        self.trigrams = defaultdict(lambda: defaultdict(lambda: defaultdict(float)))
        self.positions = defaultdict(lambda: defaultdict(lambda: defaultdict(float)))
    
    def train(self, words: List[str]):
        """Train HMM guide."""
        print("Training HMM guide for MCTS...")
        
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
    
    def get_probs(self, pattern: str, guessed: Set[str]) -> Dict[str, float]:
        """Get guide probabilities."""
        available = set(self.alphabet) - guessed
        scores = defaultdict(float)
        length = len(pattern)
        
        # Position-based
        if length in self.positions:
            for i, char in enumerate(pattern):
                if char == '_':
                    for letter in available:
                        scores[letter] += self.positions[length][i].get(letter, 0) * 3.0
        
        # Context
        revealed = [c for c in pattern if c != '_']
        
        for letter in available:
            scores[letter] += self.unigrams.get(letter, 0) * 1.0
        
        if len(revealed) >= 1:
            last = revealed[-1]
            for letter in available:
                scores[letter] += self.bigrams[last][letter] * 2.0
        
        if len(revealed) >= 2:
            prev2, prev1 = revealed[-2], revealed[-1]
            for letter in available:
                scores[letter] += self.trigrams[prev2][prev1][letter] * 2.5
        
        # Normalize
        total = sum(scores.values())
        if total > 0:
            return {l: scores[l] / total for l in available}
        return {l: 1.0 / len(available) for l in available}


class MCTSNode:
    """Node in Monte Carlo Tree Search."""
    
    def __init__(self, pattern: str, guessed: Set[str], wrong: int, parent: Optional['MCTSNode'] = None, action: Optional[str] = None):
        self.pattern = pattern
        self.guessed = guessed.copy()
        self.wrong = wrong
        self.parent = parent
        self.action = action  # Action that led to this node
        
        self.children = {}  # action -> node
        self.visits = 0
        self.value_sum = 0.0
        self.untried_actions = None
    
    def is_terminal(self) -> bool:
        """Check if terminal state."""
        return '_' not in self.pattern or self.wrong >= 6
    
    def is_fully_expanded(self) -> bool:
        """Check if all actions have been tried."""
        if self.untried_actions is None:
            available = set('abcdefghijklmnopqrstuvwxyz') - self.guessed
            self.untried_actions = list(available)
        return len(self.untried_actions) == 0
    
    def best_child(self, c: float = 1.41) -> 'MCTSNode':
        """Select best child using UCB1."""
        best_score = -float('inf')
        best_child = None
        
        for child in self.children.values():
            if child.visits == 0:
                score = float('inf')
            else:
                # UCB1 formula
                exploitation = child.value_sum / child.visits
                exploration = c * math.sqrt(math.log(self.visits) / child.visits)
                score = exploitation + exploration
            
            if score > best_score:
                best_score = score
                best_child = child
        
        return best_child
    
    def expand(self, hmm: HMMGuide) -> 'MCTSNode':
        """Expand tree by trying an untried action."""
        if self.untried_actions is None:
            available = set('abcdefghijklmnopqrstuvwxyz') - self.guessed
            self.untried_actions = list(available)
        
        if not self.untried_actions:
            return self
        
        # Use HMM to guide action selection
        hmm_probs = hmm.get_probs(self.pattern, self.guessed)
        
        # Weight untried actions by HMM probabilities
        action_weights = [hmm_probs.get(a, 0) for a in self.untried_actions]
        total_weight = sum(action_weights)
        if total_weight > 0:
            action_weights = [w / total_weight for w in action_weights]
            action = random.choices(self.untried_actions, weights=action_weights)[0]
        else:
            action = random.choice(self.untried_actions)
        
        self.untried_actions.remove(action)
        
        # Create child node
        new_pattern = list(self.pattern)
        new_guessed = self.guessed.copy()
        new_guessed.add(action)
        new_wrong = self.wrong
        
        # Simulate taking action (we don't know the word, so estimate)
        # This is a simplification - in real MCTS for Hangman, you'd need word candidates
        # For now, assume action reveals something with HMM probability
        hmm_prob = hmm_probs.get(action, 0)
        if random.random() < hmm_prob:
            # Assume it reveals a letter (simplified)
            new_wrong = self.wrong
        else:
            new_wrong = self.wrong + 1
        
        child = MCTSNode(''.join(new_pattern), new_guessed, new_wrong, parent=self, action=action)
        self.children[action] = child
        return child
    
    def simulate(self, hmm: HMMGuide, word: str) -> float:
        """Simulate a rollout from this state."""
        pattern = list(self.pattern)
        guessed = self.guessed.copy()
        wrong = self.wrong
        reward = 0
        
        while '_' in pattern and wrong < 6:
            available = set('abcdefghijklmnopqrstuvwxyz') - guessed
            if not available:
                break
            
            # HMM-guided rollout
            hmm_probs = hmm.get_probs(''.join(pattern), guessed)
            letters = list(hmm_probs.keys())
            weights = [hmm_probs[l] for l in letters]
            action = random.choices(letters, weights=weights)[0]
            
            guessed.add(action)
            
            if action in word:
                count = sum(1 for i, c in enumerate(word) if c == action and pattern[i] == '_')
                for i, c in enumerate(word):
                    if c == action:
                        pattern[i] = action
                reward += 10 + (count * 3)
            else:
                wrong += 1
                reward -= 15
        
        # Terminal reward
        if '_' not in pattern:
            reward += 100
        else:
            reward -= 80
        
        return reward
    
    def backpropagate(self, value: float):
        """Backpropagate value through tree."""
        node = self
        while node is not None:
            node.visits += 1
            node.value_sum += value
            node = node.parent


class MCTSAgent:
    """MCTS agent with HMM guidance."""
    
    def __init__(self, hmm: HMMGuide, simulations_per_move: int = 50, exploration_constant: float = 1.41):
        self.hmm = hmm
        self.simulations = simulations_per_move
        self.c = exploration_constant
        self.alphabet = 'abcdefghijklmnopqrstuvwxyz'
    
    def search(self, pattern: str, guessed: Set[str], wrong: int, word: str) -> str:
        """Perform MCTS search."""
        root = MCTSNode(pattern, guessed, wrong)
        
        for _ in range(self.simulations):
            node = root
            
            # Selection
            while not node.is_terminal() and node.is_fully_expanded():
                node = node.best_child(self.c)
            
            # Expansion
            if not node.is_terminal():
                node = node.expand(self.hmm)
            
            # Simulation
            value = node.simulate(self.hmm, word)
            
            # Backpropagation
            node.backpropagate(value)
        
        # Choose best action
        if root.children:
            best_action = max(root.children.items(), 
                            key=lambda x: x[1].visits if x[1].visits > 0 else -1)[0]
            return best_action
        
        # Fallback to HMM
        hmm_probs = self.hmm.get_probs(pattern, guessed)
        return max(hmm_probs.items(), key=lambda x: x[1])[0]


class SimplifiedMCTSAgent:
    """Simplified MCTS for evaluation (without knowing the word)."""
    
    def __init__(self, hmm: HMMGuide, epsilon: float = 0.05, hmm_weight: float = 0.8):
        self.hmm = hmm
        self.epsilon = epsilon
        self.hmm_weight = hmm_weight
        self.alphabet = 'abcdefghijklmnopqrstuvwxyz'
        
        # Value estimates
        self.q_values = defaultdict(lambda: defaultdict(float))
        self.visit_counts = defaultdict(lambda: defaultdict(int))
    
    def get_state_key(self, pattern: str, guessed: Set[str], wrong: int) -> str:
        """Get state key."""
        blanks = pattern.count('_')
        length = len(pattern)
        revealed = ''.join(['1' if c != '_' else '0' for c in pattern[:min(6, length)]])
        return f"{length}_{blanks}_{wrong}_{len(guessed)}_{revealed}"
    
    def choose_action(self, pattern: str, guessed: Set[str], wrong: int, training: bool = True) -> str:
        """Choose action."""
        available = set(self.alphabet) - guessed
        if not available:
            return random.choice(list(self.alphabet))
        
        # Epsilon-greedy
        if training and random.random() < self.epsilon:
            hmm_probs = self.hmm.get_probs(pattern, guessed)
            letters = list(hmm_probs.keys())
            weights = [hmm_probs[l] for l in letters]
            return random.choices(letters, weights=weights)[0]
        
        # Get HMM probs
        hmm_probs = self.hmm.get_probs(pattern, guessed)
        
        # Get Q-values with UCB bonus
        state_key = self.get_state_key(pattern, guessed, wrong)
        total_visits = sum(self.visit_counts[state_key].values())
        
        scores = {}
        for letter in available:
            hmm_score = hmm_probs.get(letter, 0)
            
            # Q-value with UCB exploration bonus
            q_val = self.q_values[state_key][letter]
            visits = self.visit_counts[state_key][letter]
            
            if training and total_visits > 0 and visits > 0:
                ucb_bonus = math.sqrt(2 * math.log(total_visits) / visits)
            else:
                ucb_bonus = 0
            
            # Combine HMM + Q-value + UCB
            scores[letter] = self.hmm_weight * hmm_score + (1 - self.hmm_weight) * (q_val + ucb_bonus)
        
        return max(scores.items(), key=lambda x: x[1])[0]
    
    def update(self, state: Tuple, action: str, reward: float):
        """Update Q-values."""
        pattern, guessed, wrong = state
        state_key = self.get_state_key(pattern, guessed, wrong)
        
        # Incremental average
        self.visit_counts[state_key][action] += 1
        n = self.visit_counts[state_key][action]
        old_q = self.q_values[state_key][action]
        self.q_values[state_key][action] = old_q + (reward - old_q) / n


def train_agent(agent: SimplifiedMCTSAgent, words: List[str], num_episodes: int = 8000):
    """Train MCTS agent."""
    print(f"\nTraining MCTS Agent for {num_episodes} episodes...")
    
    wins = []
    
    for episode in range(num_episodes):
        word = random.choice(words)
        pattern = ['_'] * len(word)
        guessed = set()
        wrong = 0
        episode_reward = 0
        
        while '_' in pattern and wrong < 6:
            state = (''.join(pattern), guessed.copy(), wrong)
            action = agent.choose_action(''.join(pattern), guessed, wrong, training=True)
            
            guessed.add(action)
            
            if action in word:
                count = sum(1 for i, c in enumerate(word) if c == action and pattern[i] == '_')
                for i, c in enumerate(word):
                    if c == action:
                        pattern[i] = action
                reward = 10 + (count * 3)
            else:
                wrong += 1
                reward = -15
            
            episode_reward += reward
            agent.update(state, action, reward)
        
        # Terminal reward
        if '_' not in pattern:
            episode_reward += 100
            wins.append(1)
        else:
            episode_reward -= 80
            wins.append(0)
        
        if (episode + 1) % 500 == 0:
            recent_wins = sum(wins[-500:])
            print(f"Episode {episode + 1}/{num_episodes} | "
                  f"Win Rate: {recent_wins/500:.2%} | "
                  f"States Visited: {len(agent.q_values)}")
    
    print("Training completed!")
    return agent


def evaluate_agent(agent: SimplifiedMCTSAgent, test_words: List[str]) -> Dict:
    """Evaluate agent."""
    print(f"\nEvaluating on {len(test_words)} test words...")
    
    wins = 0
    total_wrong = 0
    total_repeated = 0
    
    for i, word in enumerate(test_words):
        pattern = ['_'] * len(word)
        guessed = set()
        wrong = 0
        
        while '_' in pattern and wrong < 6:
            action = agent.choose_action(''.join(pattern), guessed, wrong, training=False)
            
            if action in guessed:
                total_repeated += 1
            
            guessed.add(action)
            
            if action in word:
                for j, c in enumerate(word):
                    if c == action:
                        pattern[j] = action
            else:
                wrong += 1
        
        if '_' not in pattern:
            wins += 1
        total_wrong += wrong
        
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
    print("Hangman MCTS Agent - Version 18")
    print("Monte Carlo Tree Search with UCB")
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
    hmm = HMMGuide()
    hmm.train(train_words)
    
    # Initialize agent
    agent = SimplifiedMCTSAgent(
        hmm=hmm,
        epsilon=0.05,
        hmm_weight=0.8  # 80% HMM, 20% learned
    )
    
    # Train
    agent = train_agent(agent, train_words, num_episodes=8000)
    
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
    print(f"\nStates Explored: {len(agent.q_values)}")
    print("=" * 70)


if __name__ == "__main__":
    main()
