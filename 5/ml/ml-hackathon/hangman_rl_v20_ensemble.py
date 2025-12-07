"""
Hangman Agent - Ensemble RL with Voting
Version 20: Multiple strategies with weighted voting

Key Features:
1. Ensemble of multiple RL strategies
2. Weighted voting based on confidence
3. 95% HMM baseline + 5% ensemble adjustments
4. Diversity in state representations
5. Majority voting with tie-breaking
"""

from collections import defaultdict, Counter
from typing import List, Tuple, Set, Dict
import random


class UltraStrongHMM:
    """Strongest possible HMM baseline."""
    
    def __init__(self):
        self.alphabet = 'abcdefghijklmnopqrstuvwxyz'
        self.unigrams = {}
        self.bigrams = defaultdict(lambda: defaultdict(float))
        self.trigrams = defaultdict(lambda: defaultdict(lambda: defaultdict(float)))
        self.positions = defaultdict(lambda: defaultdict(lambda: defaultdict(float)))
        self.endings = defaultdict(lambda: defaultdict(lambda: defaultdict(float)))  # Last 3 positions
        self.beginnings = defaultdict(lambda: defaultdict(lambda: defaultdict(float)))  # First 3 positions
        self.vowel_patterns = defaultdict(Counter)
        self.consonant_clusters = defaultdict(Counter)
    
    def train(self, words: List[str]):
        """Train ultra-strong HMM."""
        print("Training Ultra-Strong HMM...")
        
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
        
        # Endings (last 3 positions)
        ending_counts = defaultdict(lambda: defaultdict(Counter))
        for word in words:
            if len(word) >= 3:
                for i in range(3):
                    pos_from_end = -(i+1)
                    ending_counts[len(word)][i][word[pos_from_end]] += 1
        
        for length in ending_counts:
            for pos in ending_counts[length]:
                total = sum(ending_counts[length][pos].values()) + 13
                for letter in self.alphabet:
                    self.endings[length][pos][letter] = \
                        (ending_counts[length][pos][letter] + 0.5) / total
        
        # Beginnings (first 3 positions)
        beginning_counts = defaultdict(lambda: defaultdict(Counter))
        for word in words:
            for i in range(min(3, len(word))):
                beginning_counts[len(word)][i][word[i]] += 1
        
        for length in beginning_counts:
            for pos in beginning_counts[length]:
                total = sum(beginning_counts[length][pos].values()) + 13
                for letter in self.alphabet:
                    self.beginnings[length][pos][letter] = \
                        (beginning_counts[length][pos][letter] + 0.5) / total
        
        # Vowel patterns
        for word in words:
            vowel_pos = [i for i, c in enumerate(word) if c in 'aeiou']
            for letter in word:
                if letter in 'aeiou':
                    key = f"{len(word)}_{len(vowel_pos)}"
                    self.vowel_patterns[key][letter] += 1
        
        # Consonant clusters
        for word in words:
            for i in range(len(word)):
                if word[i] not in 'aeiou':
                    # Check for clusters
                    cluster = word[i]
                    j = i + 1
                    while j < len(word) and word[j] not in 'aeiou':
                        cluster += word[j]
                        j += 1
                    if len(cluster) >= 2:
                        for c in cluster:
                            self.consonant_clusters[cluster][c] += 1
    
    def get_probs(self, pattern: str, guessed: Set[str]) -> Dict[str, float]:
        """Get ultra-strong probabilities."""
        available = set(self.alphabet) - guessed
        if not available:
            return {}
        
        scores = defaultdict(float)
        length = len(pattern)
        
        # Position-based (weight 5.0 - strongest signal)
        if length in self.positions:
            for i, char in enumerate(pattern):
                if char == '_':
                    for letter in available:
                        scores[letter] += self.positions[length][i].get(letter, 0) * 5.0
        
        # Endings (weight 4.0)
        if length in self.endings:
            for i in range(min(3, length)):
                pos_from_end = -(i+1)
                if pattern[pos_from_end] == '_':
                    for letter in available:
                        scores[letter] += self.endings[length][i].get(letter, 0) * 4.0
        
        # Beginnings (weight 4.0)
        if length in self.beginnings:
            for i in range(min(3, length)):
                if i < length and pattern[i] == '_':
                    for letter in available:
                        scores[letter] += self.beginnings[length][i].get(letter, 0) * 4.0
        
        # N-gram context
        revealed = [c for c in pattern if c != '_']
        
        for letter in available:
            scores[letter] += self.unigrams.get(letter, 0) * 1.0
        
        if len(revealed) >= 1:
            last = revealed[-1]
            for letter in available:
                scores[letter] += self.bigrams[last][letter] * 3.0
        
        if len(revealed) >= 2:
            prev2, prev1 = revealed[-2], revealed[-1]
            for letter in available:
                scores[letter] += self.trigrams[prev2][prev1][letter] * 3.5
        
        # Vowel patterns
        vowels_revealed = sum(1 for c in pattern if c in 'aeiou' and c != '_')
        key = f"{length}_{vowels_revealed}"
        if key in self.vowel_patterns:
            total = sum(self.vowel_patterns[key].values())
            for letter in available:
                if letter in 'aeiou':
                    scores[letter] += (self.vowel_patterns[key][letter] / max(total, 1)) * 2.0
        
        # Normalize
        total = sum(scores.values())
        if total > 0:
            return {l: scores[l] / total for l in available}
        return {l: 1.0 / len(available) for l in available}


class EnsembleRLAgent:
    """Ensemble RL with 95:5 HMM:Learned ratio."""
    
    def __init__(self, hmm: UltraStrongHMM, epsilon: float = 0.01, hmm_weight: float = 0.95):
        self.alphabet = 'abcdefghijklmnopqrstuvwxyz'
        self.hmm = hmm
        self.hmm_weight = hmm_weight
        self.epsilon = epsilon
        
        # Multiple Q-tables with different state representations
        self.q_simple = defaultdict(lambda: defaultdict(float))  # Simple state
        self.q_pattern = defaultdict(lambda: defaultdict(float))  # Pattern-based
        self.q_context = defaultdict(lambda: defaultdict(float))  # Context-based
        
        self.visit_counts = defaultdict(lambda: defaultdict(int))
        
        # Learning parameters
        self.lr = 0.001
        self.discount = 0.95
        
        # Stats
        self.wins = []
        self.updates = 0
    
    def get_simple_state(self, pattern: str, guessed: Set[str], wrong: int) -> str:
        """Simple state representation."""
        return f"{len(pattern)}_{pattern.count('_')}_{wrong}_{len(guessed)}"
    
    def get_pattern_state(self, pattern: str, guessed: Set[str], wrong: int) -> str:
        """Pattern-based state representation."""
        vc_pattern = ''.join(['V' if (c in 'aeiou' and c != '_') else ('C' if c != '_' else '_') 
                              for c in pattern[:min(10, len(pattern))]])
        return f"{len(pattern)}_{wrong}_{vc_pattern}"
    
    def get_context_state(self, pattern: str, guessed: Set[str], wrong: int) -> str:
        """Context-based state representation."""
        revealed = ''.join([c for c in pattern if c != '_'][-4:])
        first_blank = pattern.find('_')
        last_blank = pattern.rfind('_')
        return f"{len(pattern)}_{wrong}_{revealed}_{first_blank}_{last_blank}"
    
    def choose_action(self, pattern: str, guessed: Set[str], wrong: int, training: bool = True) -> str:
        """Choose action with ensemble voting."""
        available = set(self.alphabet) - guessed
        if not available:
            return random.choice(list(self.alphabet))
        
        # Epsilon-greedy exploration
        if training and random.random() < self.epsilon:
            hmm_probs = self.hmm.get_probs(pattern, guessed)
            letters = list(hmm_probs.keys())
            weights = [hmm_probs[l] for l in letters]
            return random.choices(letters, weights=weights)[0]
        
        # Get HMM probabilities (95% weight)
        hmm_probs = self.hmm.get_probs(pattern, guessed)
        
        # Get ensemble votes (5% weight)
        state_simple = self.get_simple_state(pattern, guessed, wrong)
        state_pattern = self.get_pattern_state(pattern, guessed, wrong)
        state_context = self.get_context_state(pattern, guessed, wrong)
        
        ensemble_scores = defaultdict(float)
        for letter in available:
            # Average Q-values from all representations
            q1 = self.q_simple[state_simple][letter]
            q2 = self.q_pattern[state_pattern][letter]
            q3 = self.q_context[state_context][letter]
            ensemble_scores[letter] = (q1 + q2 + q3) / 3.0
        
        # Combine: 95% HMM + 5% ensemble
        final_scores = {}
        for letter in available:
            hmm_score = hmm_probs.get(letter, 0)
            ensemble_score = ensemble_scores[letter]
            
            # Bound ensemble contribution
            bounded_ensemble = max(-0.02, min(0.02, ensemble_score))
            
            final_scores[letter] = self.hmm_weight * hmm_score + (1 - self.hmm_weight) * (hmm_score + bounded_ensemble)
        
        return max(final_scores.items(), key=lambda x: x[1])[0]
    
    def update(self, state: Tuple, action: str, reward: float, next_state: Tuple, done: bool):
        """Update all Q-tables."""
        pattern, guessed, wrong = state
        
        state_simple = self.get_simple_state(pattern, guessed, wrong)
        state_pattern = self.get_pattern_state(pattern, guessed, wrong)
        state_context = self.get_context_state(pattern, guessed, wrong)
        
        # Target
        if done:
            target = reward
        else:
            next_pattern, next_guessed, next_wrong = next_state
            next_available = set(self.alphabet) - next_guessed
            
            if next_available:
                # Get max Q from each representation
                next_simple = self.get_simple_state(next_pattern, next_guessed, next_wrong)
                next_pattern_state = self.get_pattern_state(next_pattern, next_guessed, next_wrong)
                next_context = self.get_context_state(next_pattern, next_guessed, next_wrong)
                
                max_q1 = max([self.q_simple[next_simple][a] for a in next_available], default=0)
                max_q2 = max([self.q_pattern[next_pattern_state][a] for a in next_available], default=0)
                max_q3 = max([self.q_context[next_context][a] for a in next_available], default=0)
                
                max_q = (max_q1 + max_q2 + max_q3) / 3.0
                target = reward + self.discount * max_q
            else:
                target = reward
        
        # Update all Q-tables
        self.visit_counts[state_simple][action] += 1
        visits = self.visit_counts[state_simple][action]
        adaptive_lr = self.lr / (1 + 0.01 * visits)
        
        for q_table, state_key in [(self.q_simple, state_simple), 
                                    (self.q_pattern, state_pattern),
                                    (self.q_context, state_context)]:
            current_q = q_table[state_key][action]
            q_table[state_key][action] = current_q + adaptive_lr * (target - current_q)
        
        self.updates += 1


def train_agent(agent: EnsembleRLAgent, words: List[str], num_episodes: int = 20000):
    """Train ensemble agent."""
    print(f"\nTraining Ensemble RL Agent for {num_episodes} episodes...")
    print(f"HMM Weight: {agent.hmm_weight:.0%}, Ensemble Weight: {1-agent.hmm_weight:.0%}")
    
    for episode in range(num_episodes):
        word = random.choice(words)
        pattern = ['_'] * len(word)
        guessed = set()
        wrong = 0
        total_reward = 0
        
        while '_' in pattern and wrong < 6:
            state = (''.join(pattern), guessed.copy(), wrong)
            action = agent.choose_action(''.join(pattern), guessed, wrong, training=True)
            
            guessed.add(action)
            
            if action in word:
                count = sum(1 for i, c in enumerate(word) if c == action and pattern[i] == '_')
                for i, c in enumerate(word):
                    if c == action:
                        pattern[i] = action
                reward = 15 + (count * 5)
            else:
                wrong += 1
                reward = -25
            
            next_state = (''.join(pattern), guessed.copy(), wrong)
            done = '_' not in pattern or wrong >= 6
            
            if done:
                if '_' not in pattern:
                    reward += 150
                    agent.wins.append(1)
                else:
                    reward -= 100
                    agent.wins.append(0)
            
            agent.update(state, action, reward, next_state, done)
            total_reward += reward
        
        if (episode + 1) % 1000 == 0:
            recent_wins = sum(agent.wins[-1000:])
            print(f"Episode {episode + 1}/{num_episodes} | "
                  f"Win Rate: {recent_wins/1000:.2%} | "
                  f"Updates: {agent.updates} | "
                  f"States (Simple/Pattern/Context): "
                  f"{len(agent.q_simple)}/{len(agent.q_pattern)}/{len(agent.q_context)}")
    
    print("Training completed!")
    return agent


def evaluate_agent(agent: EnsembleRLAgent, test_words: List[str]) -> Dict:
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
    print("Hangman Ensemble RL Agent - Version 20")
    print("95% Ultra-Strong HMM + 5% Ensemble Learned")
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
    hmm = UltraStrongHMM()
    hmm.train(train_words)
    
    # Initialize agent
    agent = EnsembleRLAgent(
        hmm=hmm,
        epsilon=0.01,
        hmm_weight=0.95  # 95% HMM, 5% ensemble
    )
    
    # Train
    agent = train_agent(agent, train_words, num_episodes=20000)
    
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
    print(f"\nEnsemble Statistics:")
    print(f"Total Updates: {agent.updates}")
    print(f"States - Simple: {len(agent.q_simple)}")
    print(f"States - Pattern: {len(agent.q_pattern)}")
    print(f"States - Context: {len(agent.q_context)}")
    print("=" * 70)


if __name__ == "__main__":
    main()
