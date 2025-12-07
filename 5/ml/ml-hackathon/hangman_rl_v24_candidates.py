"""
Hangman Agent - Word Candidate Filtering with RL
Version 24: Maintain candidate set + RL letter selection

Breakthrough Architecture:
1. Track possible words matching revealed pattern
2. RL learns which letters best discriminate candidates
3. Information gain + RL confidence scores
4. Dynamic strategy based on candidate set size
5. Fallback to HMM when candidates exhausted

Target: 30%+ by combining word matching with RL strategy
"""

from collections import defaultdict, Counter
from typing import List, Tuple, Set, Dict, Optional
import random
import math


class WordCandidateFilter:
    """Maintains set of candidate words matching pattern."""
    
    def __init__(self, word_list: List[str]):
        # Organize words by length for faster lookup
        self.words_by_length = defaultdict(list)
        for word in word_list:
            self.words_by_length[len(word)].append(word)
        
        print(f"Indexed {len(word_list)} words by length")
    
    def get_candidates(self, pattern: str, guessed: Set[str]) -> List[str]:
        """Get words matching current pattern."""
        length = len(pattern)
        candidates = []
        
        for word in self.words_by_length.get(length, []):
            if self.matches_pattern(word, pattern, guessed):
                candidates.append(word)
        
        return candidates
    
    def matches_pattern(self, word: str, pattern: str, guessed: Set[str]) -> bool:
        """Check if word matches pattern and guessed constraints."""
        if len(word) != len(pattern):
            return False
        
        for i, (w_char, p_char) in enumerate(zip(word, pattern)):
            if p_char == '_':
                # Position must not be guessed
                if w_char in guessed:
                    return False
            else:
                # Position must match
                if w_char != p_char:
                    return False
        
        # All guessed letters must appear if they're in the word
        for g in guessed:
            if g in word and g not in pattern:
                return False
        
        return True
    
    def calculate_information_gain(self, candidates: List[str], letter: str) -> float:
        """Calculate expected information gain from guessing letter."""
        if not candidates:
            return 0.0
        
        # Count how many candidates have this letter
        with_letter = sum(1 for word in candidates if letter in word)
        without_letter = len(candidates) - with_letter
        
        if with_letter == 0 or without_letter == 0:
            return 0.0
        
        # Shannon entropy-based information gain
        p_with = with_letter / len(candidates)
        p_without = without_letter / len(candidates)
        
        entropy = -(p_with * math.log2(p_with) + p_without * math.log2(p_without))
        
        return entropy
    
    def get_letter_frequencies(self, candidates: List[str], guessed: Set[str]) -> Dict[str, float]:
        """Get letter frequencies in candidates."""
        freq = Counter()
        for word in candidates:
            freq.update(set(word) - guessed)
        
        total = sum(freq.values())
        if total == 0:
            return {}
        
        return {l: count / total for l, count in freq.items()}


class RLCandidateSelector:
    """RL agent that learns to select letters given candidate information."""
    
    def __init__(self, learning_rate: float = 0.015, discount: float = 0.92, epsilon: float = 0.08):
        self.lr = learning_rate
        self.gamma = discount
        self.epsilon = epsilon
        
        # Q-values for (state, letter) pairs
        self.Q = defaultdict(lambda: defaultdict(float))
        
        # Visit counts
        self.visits = defaultdict(lambda: defaultdict(int))
        
        # Stats
        self.updates = 0
        self.epsilon_decay = 0.99995
        self.epsilon_min = 0.02
    
    def get_state_key(self, num_candidates: int, pattern: str, guessed: Set[str], wrong: int) -> str:
        """Create state representation."""
        # Bin candidate counts
        if num_candidates == 0:
            cand_bin = 'none'
        elif num_candidates == 1:
            cand_bin = 'one'
        elif num_candidates <= 5:
            cand_bin = 'few'
        elif num_candidates <= 20:
            cand_bin = 'some'
        elif num_candidates <= 100:
            cand_bin = 'many'
        else:
            cand_bin = 'lots'
        
        # Pattern info
        length = len(pattern)
        blanks = pattern.count('_')
        progress = int((1 - blanks / length) * 10) if length > 0 else 0
        
        # Game state
        wrong_bin = min(wrong, 6)
        guessed_bin = len(guessed) // 3  # 0-8
        
        return f"{cand_bin}:{length}:{progress}:{wrong_bin}:{guessed_bin}"
    
    def select_letter(self, candidates: List[str], pattern: str, guessed: Set[str], 
                     wrong: int, info_gains: Dict[str, float], 
                     training: bool = True) -> str:
        """Select letter using Q-values + information gain."""
        available = set('abcdefghijklmnopqrstuvwxyz') - guessed
        if not available:
            return random.choice(list('abcdefghijklmnopqrstuvwxyz'))
        
        state_key = self.get_state_key(len(candidates), pattern, guessed, wrong)
        
        # Epsilon-greedy
        if training and random.random() < self.epsilon:
            return random.choice(list(available))
        
        # Combine Q-value with information gain
        scores = {}
        for letter in available:
            q_val = self.Q[state_key][letter]
            info_gain = info_gains.get(letter, 0)
            
            # Weighted combination
            scores[letter] = 0.6 * q_val + 0.4 * info_gain
        
        return max(scores.items(), key=lambda x: x[1])[0]
    
    def update(self, state_key: str, action: str, reward: float, 
               next_state_key: str, next_available: Set[str], done: bool):
        """Q-learning update."""
        current_q = self.Q[state_key][action]
        
        if done:
            target = reward
        else:
            if next_available:
                max_next_q = max([self.Q[next_state_key][a] for a in next_available], default=0)
            else:
                max_next_q = 0
            target = reward + self.gamma * max_next_q
        
        # Update with adaptive learning rate
        self.visits[state_key][action] += 1
        visits = self.visits[state_key][action]
        adaptive_lr = self.lr / (1 + 0.05 * math.log(1 + visits))
        
        self.Q[state_key][action] += adaptive_lr * (target - current_q)
        self.updates += 1
    
    def decay_epsilon(self):
        """Decay exploration."""
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)


class HybridRLAgent:
    """Hybrid agent combining candidates + RL."""
    
    def __init__(self, train_words: List[str]):
        self.candidate_filter = WordCandidateFilter(train_words)
        self.rl_selector = RLCandidateSelector()
        
        # Fallback HMM
        self.alphabet = 'abcdefghijklmnopqrstuvwxyz'
        self.letter_freq = Counter()
        for word in train_words:
            self.letter_freq.update(word)
        total = sum(self.letter_freq.values())
        self.letter_freq = {l: self.letter_freq[l] / total for l in self.alphabet}
        
        # Stats
        self.wins = []
        self.episode_rewards = []
    
    def choose_action(self, pattern: str, guessed: Set[str], wrong: int, training: bool = True) -> str:
        """Choose action using hybrid approach."""
        # Get candidates
        candidates = self.candidate_filter.get_candidates(pattern, guessed)
        
        available = set(self.alphabet) - guessed
        if not available:
            return random.choice(list(self.alphabet))
        
        # Calculate information gains
        if candidates:
            info_gains = {}
            for letter in available:
                info_gains[letter] = self.candidate_filter.calculate_information_gain(candidates, letter)
            
            # Also consider frequency in candidates
            letter_freqs = self.candidate_filter.get_letter_frequencies(candidates, guessed)
            for letter in available:
                if letter in letter_freqs:
                    info_gains[letter] += letter_freqs[letter] * 2.0
        else:
            # Fallback: use global frequency
            info_gains = {l: self.letter_freq.get(l, 0) for l in available}
        
        # Use RL selector
        return self.rl_selector.select_letter(candidates, pattern, guessed, wrong, 
                                              info_gains, training)
    
    def update(self, state_before: Tuple, action: str, reward: float, 
               state_after: Optional[Tuple], done: bool):
        """Update RL."""
        pattern_before, guessed_before, wrong_before = state_before
        candidates_before = self.candidate_filter.get_candidates(pattern_before, guessed_before)
        state_key_before = self.rl_selector.get_state_key(
            len(candidates_before), pattern_before, guessed_before, wrong_before)
        
        if state_after and not done:
            pattern_after, guessed_after, wrong_after = state_after
            candidates_after = self.candidate_filter.get_candidates(pattern_after, guessed_after)
            state_key_after = self.rl_selector.get_state_key(
                len(candidates_after), pattern_after, guessed_after, wrong_after)
            next_available = set(self.alphabet) - guessed_after
        else:
            state_key_after = ""
            next_available = set()
        
        self.rl_selector.update(state_key_before, action, reward, state_key_after, 
                               next_available, done)


def train_agent(agent: HybridRLAgent, train_words: List[str], num_episodes: int = 35000):
    """Train hybrid agent."""
    print(f"\nTraining Hybrid RL Agent for {num_episodes} episodes...")
    
    for episode in range(num_episodes):
        word = random.choice(train_words)
        pattern = ['_'] * len(word)
        guessed = set()
        wrong = 0
        total_reward = 0
        
        while '_' in pattern and wrong < 6:
            state = (''.join(pattern), guessed.copy(), wrong)
            action = agent.choose_action(''.join(pattern), guessed, wrong, training=True)
            
            was_correct = action in word
            guessed.add(action)
            
            if was_correct:
                revealed = sum(1 for i, c in enumerate(word) if c == action and pattern[i] == '_')
                for i, c in enumerate(word):
                    if c == action:
                        pattern[i] = action
                reward = 20 + revealed * 12
            else:
                wrong += 1
                reward = -40
            
            next_state = (''.join(pattern), guessed.copy(), wrong)
            done = '_' not in pattern or wrong >= 6
            
            if done:
                if '_' not in pattern:
                    reward += 300
                else:
                    reward -= 250
            
            agent.update(state, action, reward, next_state if not done else None, done)
            total_reward += reward
        
        agent.wins.append(1 if '_' not in pattern else 0)
        agent.episode_rewards.append(total_reward)
        agent.rl_selector.decay_epsilon()
        
        if (episode + 1) % 2000 == 0:
            recent_wins = sum(agent.wins[-2000:])
            avg_reward = sum(agent.episode_rewards[-2000:]) / 2000
            print(f"Episode {episode + 1}/{num_episodes} | "
                  f"Win Rate: {recent_wins/2000:.2%} | "
                  f"Avg Reward: {avg_reward:.1f} | "
                  f"Epsilon: {agent.rl_selector.epsilon:.4f} | "
                  f"Q-Updates: {agent.rl_selector.updates} | "
                  f"States: {len(agent.rl_selector.Q)}")
    
    print("Training Complete!")
    return agent


def evaluate_agent(agent: HybridRLAgent, test_words: List[str]) -> Dict:
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
        
        if (i + 1) % 250 == 0:
            print(f"Progress: {i + 1}/{len(test_words)} | Win Rate: {wins/(i+1):.2%}")
    
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
    print("=" * 80)
    print("Hangman RL V24 - Word Candidate Filtering + RL")
    print("Information Gain + Q-Learning Strategy")
    print("=" * 80)
    
    # Load data
    with open('data/corpus.txt', 'r') as f:
        train_words = [line.strip().lower() for line in f if line.strip()]
    print(f"Loaded {len(train_words)} training words")
    
    with open('data/test.txt', 'r') as f:
        test_words = [line.strip().lower() for line in f if line.strip()]
    print(f"Loaded {len(test_words)} test words")
    
    # Create agent
    agent = HybridRLAgent(train_words)
    
    # Train
    agent = train_agent(agent, train_words, num_episodes=35000)
    
    # Evaluate
    results = evaluate_agent(agent, test_words)
    
    # Results
    print("\n" + "=" * 80)
    print("FINAL RESULTS")
    print("=" * 80)
    print(f"Success Rate: {results['success_rate']:.4f} ({results['wins']}/{len(test_words)})")
    print(f"Average Wrong: {results['avg_wrong']:.4f}")
    print(f"Average Repeated: {results['avg_repeated']:.4f}")
    print(f"Total Wrong: {results['total_wrong']}")
    print(f"Total Repeated: {results['total_repeated']}")
    print(f"\nFinal Score: {results['final_score']:.2f}")
    print(f"\nRL Statistics:")
    print(f"Q-Updates: {agent.rl_selector.updates}")
    print(f"States Learned: {len(agent.rl_selector.Q)}")
    print(f"Final Epsilon: {agent.rl_selector.epsilon:.4f}")
    print("=" * 80)
    
    print("\n" + "=" * 80)
    print("RL COMPONENTS")
    print("=" * 80)
    print("✓ Exploration: Epsilon-greedy with decay")
    print("✓ Learning: Q-learning on letter selection")
    print("✓ Rewards: Performance-based feedback")
    print("✓ State space: Candidate count + pattern + game state")
    print("✓ Policy: Q-values + Information gain")
    print("=" * 80)


if __name__ == "__main__":
    main()
