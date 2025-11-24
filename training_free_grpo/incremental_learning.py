"""
Incremental Learning Module

Enables continuous learning from all rollouts, not just partially correct ones.

Key features:
1. Online experience updates (no need to wait for batch)
2. Learn from all outcomes: success, failure, partial success
3. Fast adaptation to new problem types
4. Contrastive learning: what works vs what doesn't
"""

import json
import os
from typing import List, Dict, Tuple
from collections import deque
import time
from training_free_grpo.llm import LLM


class IncrementalExperienceUpdater:
    """
    Updates experiences incrementally as new rollouts complete.

    Unlike batch updates, this enables:
    - Immediate learning from successful solutions
    - Fast identification of common pitfalls
    - Adaptive strategies based on recent performance
    """

    def __init__(
        self,
        window_size: int = 100,
        update_frequency: int = 10,
        save_path: str = "incremental_experiences.json"
    ):
        """
        Args:
            window_size: Number of recent rollouts to consider
            update_frequency: Update experiences every N rollouts
            save_path: Path to save incremental experiences
        """
        self.llm = LLM()
        self.window_size = window_size
        self.update_frequency = update_frequency
        self.save_path = save_path

        # Sliding window of recent rollouts
        self.recent_successes = deque(maxlen=window_size)
        self.recent_failures = deque(maxlen=window_size)

        # Incremental experiences
        self.experiences = {}
        self.load()

        # Counters
        self.rollout_count = 0
        self.last_update_time = time.time()

    def load(self):
        """Load incremental experiences from file."""
        if os.path.exists(self.save_path):
            with open(self.save_path, 'r') as f:
                data = json.load(f)
                self.experiences = data.get("experiences", {})
                self.rollout_count = data.get("rollout_count", 0)
                print(f"[Incremental] Loaded {len(self.experiences)} experiences "
                      f"from {self.rollout_count} rollouts")

    def save(self):
        """Save incremental experiences to file."""
        data = {
            "experiences": self.experiences,
            "rollout_count": self.rollout_count,
            "last_updated": time.time()
        }
        with open(self.save_path, 'w') as f:
            json.dump(data, f, indent=2)

    def add_rollout(
        self,
        problem: str,
        response: str,
        reward: float,
        trajectory: List[Dict] = None
    ):
        """
        Add a single rollout and potentially trigger an update.

        Args:
            problem: Problem text
            response: Agent's response
            reward: Reward obtained (0 or 1)
            trajectory: Optional full trajectory with tool calls
        """
        rollout_data = {
            "problem": problem,
            "response": response,
            "reward": reward,
            "trajectory": trajectory,
            "timestamp": time.time()
        }

        if reward > 0.5:
            self.recent_successes.append(rollout_data)
        else:
            self.recent_failures.append(rollout_data)

        self.rollout_count += 1

        # Check if it's time to update
        if self.rollout_count % self.update_frequency == 0:
            self._trigger_update()

    def _trigger_update(self):
        """Trigger an incremental experience update."""
        print(f"\n[Incremental] Triggering update at rollout {self.rollout_count}")
        print(f"  Recent successes: {len(self.recent_successes)}")
        print(f"  Recent failures: {len(self.recent_failures)}")

        if len(self.recent_successes) < 2 and len(self.recent_failures) < 2:
            print("  → Insufficient data, skipping")
            return

        # Extract patterns from recent rollouts
        new_experiences = self._extract_patterns()

        # Merge with existing experiences
        self._merge_experiences(new_experiences)

        # Save
        self.save()

        update_time = time.time() - self.last_update_time
        print(f"  → Update complete in {update_time:.1f}s, "
              f"now have {len(self.experiences)} experiences")
        self.last_update_time = time.time()

    def _extract_patterns(self) -> List[str]:
        """
        Extract experience patterns from recent rollouts using contrastive learning.

        Compares successful vs unsuccessful attempts to identify key differences.
        """
        if len(self.recent_successes) == 0 and len(self.recent_failures) == 0:
            return []

        # Sample recent rollouts for analysis
        success_samples = list(self.recent_successes)[-5:]  # Last 5 successes
        failure_samples = list(self.recent_failures)[-5:]  # Last 5 failures

        prompt = self._build_contrastive_prompt(success_samples, failure_samples)

        try:
            response = self.llm.chat(prompt, temperature=0.3, max_tokens=1024)
            # Extract experiences from response
            experiences = self._parse_experiences(response)
            return experiences
        except Exception as e:
            print(f"Warning: Pattern extraction failed: {e}")
            return []

    def _build_contrastive_prompt(
        self,
        successes: List[Dict],
        failures: List[Dict]
    ) -> str:
        """Build prompt for contrastive pattern extraction."""

        # Format successful attempts
        success_text = ""
        if successes:
            success_text = "SUCCESSFUL ATTEMPTS:\n"
            for i, rollout in enumerate(successes, 1):
                success_text += f"\n{i}. Problem: {rollout['problem'][:150]}...\n"
                success_text += f"   Solution approach: {rollout['response'][:200]}...\n"

        # Format failed attempts
        failure_text = ""
        if failures:
            failure_text = "\nFAILED ATTEMPTS:\n"
            for i, rollout in enumerate(failures, 1):
                failure_text += f"\n{i}. Problem: {rollout['problem'][:150]}...\n"
                failure_text += f"   Attempted approach: {rollout['response'][:200]}...\n"

        prompt = f"""
You are analyzing problem-solving patterns to extract generalizable experiences.

{success_text}

{failure_text}

TASK: Compare successful vs failed attempts and extract 2-3 key insights.

Focus on:
1. What strategies consistently work across successful attempts?
2. What mistakes or approaches appear in failures but not successes?
3. What general principles can guide future problem-solving?

Requirements:
- Each insight should be 1-2 sentences
- Must be GENERAL (not problem-specific)
- Must be ACTIONABLE (provide clear guidance)
- Use conditional phrasing: "When X, do Y" or "To achieve X, use Y"

Format your response as:
INSIGHT 1: [insight text]
INSIGHT 2: [insight text]
INSIGHT 3: [insight text]
"""

        return prompt

    def _parse_experiences(self, response: str) -> List[str]:
        """Parse experiences from LLM response."""
        experiences = []
        lines = response.strip().split('\n')

        for line in lines:
            line = line.strip()
            if line.startswith(('INSIGHT', '1.', '2.', '3.', '-', '*')):
                # Extract insight text
                if ':' in line:
                    text = line.split(':', 1)[1].strip()
                else:
                    text = line.lstrip('123.-* ').strip()

                if len(text) > 20 and len(text) < 300:  # Reasonable length
                    experiences.append(text)

        return experiences

    def _merge_experiences(self, new_experiences: List[str]):
        """
        Merge new experiences with existing ones.

        Strategy:
        - Add truly novel experiences
        - Update similar existing ones to be more general
        - Remove outdated experiences with low effectiveness
        """
        for new_exp in new_experiences:
            # Check if similar experience exists
            similar_id = self._find_similar_experience(new_exp)

            if similar_id:
                # Update existing experience to be more general
                self._generalize_experience(similar_id, new_exp)
            else:
                # Add as new experience
                new_id = f"I{len(self.experiences)}"  # 'I' for Incremental
                self.experiences[new_id] = new_exp
                print(f"  → Added {new_id}: {new_exp}")

    def _find_similar_experience(self, new_exp: str, threshold: float = 0.6) -> str:
        """Find if a similar experience already exists."""
        def tokenize(text: str) -> set:
            import re
            return set(re.findall(r'\w+', text.lower()))

        new_tokens = tokenize(new_exp)

        for exp_id, exp_text in self.experiences.items():
            exp_tokens = tokenize(exp_text)
            intersection = len(new_tokens & exp_tokens)
            union = len(new_tokens | exp_tokens)
            similarity = intersection / union if union > 0 else 0.0

            if similarity >= threshold:
                return exp_id

        return None

    def _generalize_experience(self, exp_id: str, new_exp: str):
        """
        Generalize an existing experience by combining it with a new similar one.
        """
        old_exp = self.experiences[exp_id]

        prompt = f"""
Combine these two similar experiences into one more general experience:

EXPERIENCE 1: {old_exp}
EXPERIENCE 2: {new_exp}

Create a single experience that:
1. Captures the core idea of both
2. Is more general than either individual one
3. Remains concise (1-2 sentences)

Respond with ONLY the combined experience text.
"""

        try:
            combined = self.llm.chat(prompt, temperature=0.0, max_tokens=256)
            self.experiences[exp_id] = combined.strip()
            print(f"  → Updated {exp_id}: {combined}")
        except Exception as e:
            print(f"Warning: Generalization failed: {e}")

    def get_experiences(self) -> Dict[str, str]:
        """Get current incremental experiences."""
        return self.experiences.copy()

    def reset(self):
        """Reset incremental learning state."""
        self.recent_successes.clear()
        self.recent_failures.clear()
        self.rollout_count = 0
        print("[Incremental] Reset learning state")


class FastAdaptationModule:
    """
    Enables fast adaptation to new problem types.

    When encountering a new type of problem:
    1. Quickly identifies it's different from past problems
    2. Generates targeted experiences for this type
    3. Monitors performance and adapts strategies
    """

    def __init__(self):
        self.llm = LLM()
        self.problem_type_experiences = {}  # {type: [experiences]}
        self.problem_type_detector = ProblemTypeDetector()

    def adapt_to_problem(
        self,
        problem: str,
        recent_attempts: List[Tuple[str, float]]  # (response, reward)
    ) -> List[str]:
        """
        Quickly adapt to a new problem by analyzing recent attempts.

        Args:
            problem: The problem text
            recent_attempts: List of (response, reward) from recent attempts

        Returns:
            List of fast-adapted experiences specific to this problem type
        """
        # Detect problem type
        problem_type = self.problem_type_detector.detect_type(problem)

        # Check if we've seen this type before
        if problem_type in self.problem_type_experiences:
            print(f"[Fast Adapt] Using cached experiences for type: {problem_type}")
            return self.problem_type_experiences[problem_type]

        # Generate type-specific experiences
        print(f"[Fast Adapt] New problem type detected: {problem_type}")
        experiences = self._generate_type_specific_experiences(
            problem,
            problem_type,
            recent_attempts
        )

        # Cache for future
        self.problem_type_experiences[problem_type] = experiences

        return experiences

    def _generate_type_specific_experiences(
        self,
        problem: str,
        problem_type: str,
        recent_attempts: List[Tuple[str, float]]
    ) -> List[str]:
        """Generate experiences targeted at a specific problem type."""

        attempts_text = ""
        for i, (response, reward) in enumerate(recent_attempts, 1):
            status = "✓ Success" if reward > 0.5 else "✗ Failure"
            attempts_text += f"\nAttempt {i} ({status}):\n{response[:200]}...\n"

        prompt = f"""
You are creating targeted problem-solving strategies for a NEW type of problem.

PROBLEM TYPE: {problem_type}

EXAMPLE PROBLEM:
{problem}

RECENT ATTEMPTS ON SIMILAR PROBLEMS:
{attempts_text}

TASK: Create 3 specific strategies tailored to solving {problem_type} problems.

These strategies should:
1. Be SPECIFIC to {problem_type} problems (not too general)
2. Address common pitfalls seen in failed attempts
3. Build on successful patterns from successful attempts
4. Be immediately actionable

Format:
STRATEGY 1: [specific strategy]
STRATEGY 2: [specific strategy]
STRATEGY 3: [specific strategy]
"""

        try:
            response = self.llm.chat(prompt, temperature=0.3, max_tokens=1024)
            experiences = self._parse_strategies(response)
            return experiences
        except Exception as e:
            print(f"Warning: Fast adaptation failed: {e}")
            return []

    def _parse_strategies(self, response: str) -> List[str]:
        """Parse strategies from LLM response."""
        strategies = []
        lines = response.strip().split('\n')

        for line in lines:
            line = line.strip()
            if line.startswith(('STRATEGY', '1.', '2.', '3.', '-')):
                if ':' in line:
                    text = line.split(':', 1)[1].strip()
                else:
                    text = line.lstrip('123.-* ').strip()

                if len(text) > 20:
                    strategies.append(text)

        return strategies


class ProblemTypeDetector:
    """Detects the type of a problem to enable targeted strategies."""

    def __init__(self):
        self.known_types = {
            # Math types
            "optimization": ["maximize", "minimize", "optimal", "maximum", "minimum"],
            "proof": ["prove", "show that", "demonstrate", "justify"],
            "calculation": ["calculate", "compute", "find the value"],
            "equation_solving": ["solve", "find x", "find the solution"],

            # Web types
            "fact_finding": ["what is", "who is", "when did", "where is"],
            "comparison": ["compare", "difference between", "versus"],
            "explanation": ["explain", "why does", "how does"],
            "procedure": ["how to", "steps to", "process of"],
        }

    def detect_type(self, problem: str) -> str:
        """
        Detect problem type based on keywords and patterns.

        Returns:
            String indicating problem type, or "general" if no specific type detected
        """
        problem_lower = problem.lower()

        # Count matches for each type
        type_scores = {}
        for ptype, keywords in self.known_types.items():
            score = sum(1 for kw in keywords if kw in problem_lower)
            if score > 0:
                type_scores[ptype] = score

        if type_scores:
            # Return type with highest score
            best_type = max(type_scores.items(), key=lambda x: x[1])[0]
            return best_type
        else:
            return "general"


if __name__ == "__main__":
    # Example usage of incremental learning
    print("=== Incremental Learning Demo ===\n")

    updater = IncrementalExperienceUpdater(
        window_size=10,
        update_frequency=5,
        save_path="demo_incremental.json"
    )

    # Simulate some rollouts
    problems = [
        ("Find the derivative of f(x) = x^2 + 3x", "f'(x) = 2x + 3", 1.0),
        ("Find the derivative of g(x) = x^3", "g'(x) = 3x^2", 1.0),
        ("Find the derivative of h(x) = sin(x)", "h'(x) = sin(x)", 0.0),  # Wrong!
        ("Find the derivative of f(x) = cos(x)", "f'(x) = -sin(x)", 1.0),
        ("Find the derivative of f(x) = e^x", "f'(x) = e^x", 1.0),
        ("Find the derivative of f(x) = ln(x)", "f'(x) = x", 0.0),  # Wrong!
    ]

    for problem, response, reward in problems:
        updater.add_rollout(problem, response, reward)
        print(f"Added rollout: reward={reward}")

    print(f"\n=== Final Incremental Experiences ===")
    experiences = updater.get_experiences()
    for exp_id, exp_text in experiences.items():
        print(f"{exp_id}: {exp_text}")

    # Example of fast adaptation
    print("\n\n=== Fast Adaptation Demo ===\n")

    adapter = FastAdaptationModule()

    new_problem = "Prove that the derivative of x^n is n*x^(n-1)"
    recent_attempts = [
        ("Used induction but didn't establish base case", 0.0),
        ("Applied power rule directly without justification", 0.5),
        ("Used limit definition and algebraic manipulation", 1.0),
    ]

    adapted_exp = adapter.adapt_to_problem(new_problem, recent_attempts)

    print("Adapted experiences:")
    for exp in adapted_exp:
        print(f"  - {exp}")
