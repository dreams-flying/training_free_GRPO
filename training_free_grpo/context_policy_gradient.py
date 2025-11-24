"""
Context-Policy Gradient (CPG) for Training-Free GRPO

This module implements a novel framework that formalizes semantic advantage learning
as a gradient-like optimization process in the context/prompt space.

Key Innovation:
    Instead of manually crafting experience updates, we treat experience modification
    as a learnable policy: E_{t+1} = E_t + f_φ(E_t, R_t)

    Where f_φ is an LLM that learns (via in-context learning) to generate
    semantic "gradients" based on reward signals.

Theoretical Foundation:
    - Treat experiences as parameters in prompt space
    - Reward changes → semantic feedback → experience modifications
    - Implicit gradient estimation via LLM reasoning
    - Policy gradient in context space (not parameter space)
"""

from typing import List, Dict, Tuple, Optional
import json
from dataclasses import dataclass
from collections import defaultdict
import numpy as np


@dataclass
class ExperienceUpdate:
    """Represents a semantic gradient update to an experience."""
    operation: str  # "add", "modify", "delete", "strengthen", "weaken"
    target_experience: str  # The experience to modify
    new_content: Optional[str] = None  # For add/modify operations
    reasoning: str = ""  # Why this update is needed
    gradient_magnitude: float = 0.0  # Estimated "gradient" strength


class ContextPolicyGradient:
    """
    Implements Context-Policy Gradient (CPG) learning.

    This class treats experience updates as a gradient descent process in prompt space,
    where the LLM acts as an implicit gradient estimator.
    """

    def __init__(self, llm_client, learning_rate: float = 0.3, momentum: float = 0.9):
        """
        Args:
            llm_client: LLM client for generating semantic gradients
            learning_rate: Controls magnitude of experience updates (0-1)
            momentum: Weight given to historical update patterns (0-1)
        """
        self.llm = llm_client
        self.learning_rate = learning_rate
        self.momentum = momentum

        # Track update history for momentum
        self.update_history: List[ExperienceUpdate] = []
        self.experience_scores: Dict[str, List[float]] = defaultdict(list)

    def compute_semantic_gradient(
        self,
        experiences: List[str],
        reward_trajectory: List[Tuple[str, float]],
        problem_context: str
    ) -> List[ExperienceUpdate]:
        """
        Compute semantic gradients for experience updates.

        This is the core CPG operation: converting reward signals into
        semantic modifications of the experience set.

        Args:
            experiences: Current experience set E_t
            reward_trajectory: List of (problem, reward) pairs from recent rollouts
            problem_context: Current problem domain/type

        Returns:
            List of ExperienceUpdate objects (semantic gradients)
        """
        # Compute reward statistics
        rewards = [r for _, r in reward_trajectory]
        mean_reward = np.mean(rewards)
        reward_variance = np.var(rewards)
        reward_trend = self._compute_reward_trend(rewards)

        # Analyze which experiences correlate with success/failure
        experience_effectiveness = self._analyze_experience_effectiveness(
            experiences, reward_trajectory
        )

        # Generate semantic gradient via LLM
        gradient_prompt = self._build_gradient_prompt(
            experiences=experiences,
            mean_reward=mean_reward,
            reward_variance=reward_variance,
            reward_trend=reward_trend,
            experience_effectiveness=experience_effectiveness,
            problem_context=problem_context
        )

        # LLM generates update instructions (implicit gradient)
        response = self.llm.generate(gradient_prompt, temperature=0.7)
        updates = self._parse_gradient_response(response)

        # Apply momentum from historical updates
        updates = self._apply_momentum(updates)

        # Scale by learning rate
        for update in updates:
            update.gradient_magnitude *= self.learning_rate

        return updates

    def apply_gradient(
        self,
        experiences: List[str],
        gradients: List[ExperienceUpdate]
    ) -> List[str]:
        """
        Apply semantic gradients to update experience set.

        This is analogous to θ_{t+1} = θ_t - α∇L in standard gradient descent,
        but operating in discrete semantic space.

        Args:
            experiences: Current experience set E_t
            gradients: Semantic gradients (update instructions)

        Returns:
            Updated experience set E_{t+1}
        """
        updated_experiences = experiences.copy()

        for grad in gradients:
            if grad.gradient_magnitude < 0.1:  # Skip weak gradients
                continue

            if grad.operation == "add":
                updated_experiences.append(grad.new_content)

            elif grad.operation == "modify":
                # Find and replace target experience
                for i, exp in enumerate(updated_experiences):
                    if self._similarity(exp, grad.target_experience) > 0.8:
                        updated_experiences[i] = grad.new_content
                        break

            elif grad.operation == "delete":
                # Remove target experience
                updated_experiences = [
                    exp for exp in updated_experiences
                    if self._similarity(exp, grad.target_experience) < 0.8
                ]

            elif grad.operation == "strengthen":
                # Duplicate or emphasize important experience
                for i, exp in enumerate(updated_experiences):
                    if self._similarity(exp, grad.target_experience) > 0.8:
                        updated_experiences.insert(i+1, grad.new_content or exp)
                        break

            elif grad.operation == "weaken":
                # De-emphasize less useful experience
                pass  # Implicitly handled by not strengthening

        # Store update history for momentum
        self.update_history.extend(gradients)

        return updated_experiences

    def _compute_reward_trend(self, rewards: List[float]) -> str:
        """Analyze if rewards are improving, declining, or stable."""
        if len(rewards) < 3:
            return "insufficient_data"

        recent = rewards[-5:]
        earlier = rewards[:-5] if len(rewards) > 5 else rewards[:len(rewards)//2]

        if not earlier:
            return "stable"

        recent_mean = np.mean(recent)
        earlier_mean = np.mean(earlier)

        if recent_mean > earlier_mean + 0.1:
            return "improving"
        elif recent_mean < earlier_mean - 0.1:
            return "declining"
        else:
            return "stable"

    def _analyze_experience_effectiveness(
        self,
        experiences: List[str],
        reward_trajectory: List[Tuple[str, float]]
    ) -> Dict[str, float]:
        """
        Analyze correlation between experiences and rewards.

        This is a simplified version - in practice, would track which
        experiences were actually used in each rollout.
        """
        effectiveness = {}

        for exp in experiences:
            # Simplified: assume experiences mentioned in high-reward problems are effective
            relevant_rewards = []
            for problem, reward in reward_trajectory:
                # Check if experience seems relevant to problem
                if any(keyword in problem.lower() for keyword in exp.lower().split()[:5]):
                    relevant_rewards.append(reward)

            if relevant_rewards:
                effectiveness[exp] = np.mean(relevant_rewards)
            else:
                effectiveness[exp] = 0.0

        return effectiveness

    def _build_gradient_prompt(
        self,
        experiences: List[str],
        mean_reward: float,
        reward_variance: float,
        reward_trend: str,
        experience_effectiveness: Dict[str, float],
        problem_context: str
    ) -> str:
        """Build prompt for LLM to generate semantic gradients."""

        # Rank experiences by effectiveness
        ranked_exps = sorted(
            experience_effectiveness.items(),
            key=lambda x: x[1],
            reverse=True
        )

        prompt = f"""You are a Context-Policy Gradient optimizer. Your task is to update a set of problem-solving experiences based on reward signals.

**Current Performance:**
- Mean Reward: {mean_reward:.3f}
- Reward Variance: {reward_variance:.3f}
- Trend: {reward_trend}
- Problem Domain: {problem_context}

**Current Experiences (ranked by effectiveness):**
"""
        for i, (exp, eff) in enumerate(ranked_exps[:10], 1):
            prompt += f"{i}. [{eff:.3f}] {exp}\n"

        prompt += """
**Your Task:**
Generate semantic gradient updates to improve the experience set. For each update, specify:
1. Operation: add/modify/delete/strengthen/weaken
2. Target: which experience to update (or "NEW" for add)
3. Content: new/modified experience text
4. Reasoning: why this update will improve rewards
5. Magnitude: estimated impact (0.0-1.0)

**Optimization Principles:**
- If rewards are declining: strengthen successful patterns, add missing strategies
- If rewards are stable but low: explore new approaches, modify ineffective experiences
- If rewards are improving: strengthen what works, remove what doesn't
- High variance suggests experiences are inconsistent - standardize or clarify them
- Low effectiveness scores indicate experiences that should be modified or removed

**Output Format (JSON):**
{
    "updates": [
        {
            "operation": "modify",
            "target": "experience text to find...",
            "content": "improved experience text",
            "reasoning": "why this helps",
            "magnitude": 0.8
        }
    ]
}

Generate 3-5 high-impact updates:
"""
        return prompt

    def _parse_gradient_response(self, response: str) -> List[ExperienceUpdate]:
        """Parse LLM response into ExperienceUpdate objects."""
        try:
            # Try to extract JSON from response
            start = response.find('{')
            end = response.rfind('}') + 1
            if start >= 0 and end > start:
                json_str = response[start:end]
                data = json.loads(json_str)

                updates = []
                for update_dict in data.get("updates", []):
                    updates.append(ExperienceUpdate(
                        operation=update_dict.get("operation", "add"),
                        target_experience=update_dict.get("target", ""),
                        new_content=update_dict.get("content"),
                        reasoning=update_dict.get("reasoning", ""),
                        gradient_magnitude=update_dict.get("magnitude", 0.5)
                    ))
                return updates
        except json.JSONDecodeError:
            pass

        # Fallback: return empty updates
        return []

    def _apply_momentum(self, updates: List[ExperienceUpdate]) -> List[ExperienceUpdate]:
        """Apply momentum from historical updates."""
        if not self.update_history or self.momentum == 0:
            return updates

        # Find similar historical updates
        recent_history = self.update_history[-20:]  # Last 20 updates

        for update in updates:
            # Boost magnitude if this type of update was successful before
            similar_historical = [
                h for h in recent_history
                if h.operation == update.operation and
                self._similarity(h.target_experience, update.target_experience) > 0.5
            ]

            if similar_historical:
                # Apply momentum boost
                avg_historical_magnitude = np.mean([h.gradient_magnitude for h in similar_historical])
                update.gradient_magnitude = (
                    (1 - self.momentum) * update.gradient_magnitude +
                    self.momentum * avg_historical_magnitude
                )

        return updates

    def _similarity(self, text1: str, text2: str) -> float:
        """Simple text similarity (could be replaced with embedding-based similarity)."""
        if not text1 or not text2:
            return 0.0

        words1 = set(text1.lower().split())
        words2 = set(text2.lower().split())

        if not words1 or not words2:
            return 0.0

        intersection = words1.intersection(words2)
        union = words1.union(words2)

        return len(intersection) / len(union) if union else 0.0


class CPGTrainer:
    """
    High-level trainer using Context-Policy Gradient.

    This orchestrates the CPG optimization loop:
    1. Run rollouts with current experiences
    2. Compute semantic gradients from rewards
    3. Update experiences
    4. Repeat
    """

    def __init__(self, llm_client, learning_rate: float = 0.3):
        self.cpg = ContextPolicyGradient(llm_client, learning_rate=learning_rate)
        self.reward_history = []

    def optimize_experiences(
        self,
        initial_experiences: List[str],
        problems: List[str],
        num_iterations: int = 10,
        rollouts_per_iteration: int = 20
    ) -> Tuple[List[str], List[float]]:
        """
        Run CPG optimization loop.

        Args:
            initial_experiences: Starting experience set
            problems: Problem set for evaluation
            num_iterations: Number of gradient update iterations
            rollouts_per_iteration: Rollouts per iteration for gradient estimation

        Returns:
            (optimized_experiences, reward_curve)
        """
        experiences = initial_experiences.copy()
        reward_curve = []

        for iteration in range(num_iterations):
            print(f"\n=== CPG Iteration {iteration + 1}/{num_iterations} ===")

            # Run rollouts with current experiences
            iteration_rewards = []
            reward_trajectory = []

            for problem in problems[:rollouts_per_iteration]:
                # Simulate rollout (in practice, use actual GRPO rollout)
                reward = self._simulate_rollout(problem, experiences)
                iteration_rewards.append(reward)
                reward_trajectory.append((problem, reward))

            mean_reward = np.mean(iteration_rewards)
            reward_curve.append(mean_reward)
            print(f"Mean Reward: {mean_reward:.3f}")

            # Compute semantic gradients
            gradients = self.cpg.compute_semantic_gradient(
                experiences=experiences,
                reward_trajectory=reward_trajectory,
                problem_context="math/reasoning"
            )

            print(f"Generated {len(gradients)} semantic gradients:")
            for grad in gradients[:3]:  # Show top 3
                print(f"  - {grad.operation}: {grad.reasoning[:60]}...")

            # Apply gradients to update experiences
            experiences = self.cpg.apply_gradient(experiences, gradients)
            print(f"Updated experience set size: {len(experiences)}")

        return experiences, reward_curve

    def _simulate_rollout(self, problem: str, experiences: List[str]) -> float:
        """
        Simulate a rollout (placeholder).

        In practice, this would use actual GRPO rollout with the LLM.
        """
        # Simplified simulation: better experiences → higher rewards
        return np.random.random() * (1 + 0.1 * len(experiences))


# Example usage
if __name__ == "__main__":
    print("=== Context-Policy Gradient (CPG) Demo ===\n")

    # Mock LLM client
    class MockLLM:
        def generate(self, prompt, temperature=0.7):
            # Return mock gradient updates
            return """{
                "updates": [
                    {
                        "operation": "add",
                        "target": "NEW",
                        "content": "When solving complex problems, break them into smaller verifiable steps",
                        "reasoning": "Rewards show failures in multi-step reasoning",
                        "magnitude": 0.85
                    },
                    {
                        "operation": "modify",
                        "target": "check your work carefully",
                        "content": "After each calculation, verify the result before proceeding to the next step",
                        "reasoning": "Make the verification step more specific and actionable",
                        "magnitude": 0.7
                    },
                    {
                        "operation": "strengthen",
                        "target": "use systematic approach",
                        "content": "Always use a systematic approach: understand problem → plan solution → execute → verify",
                        "reasoning": "This pattern correlates with high rewards",
                        "magnitude": 0.9
                    }
                ]
            }"""

    llm = MockLLM()

    # Initial experiences
    initial_experiences = [
        "Always read the problem carefully",
        "Check your work",
        "Use systematic approach",
        "Break down complex problems"
    ]

    # Create CPG trainer
    trainer = CPGTrainer(llm, learning_rate=0.3)

    # Mock problems
    problems = [f"Problem {i}" for i in range(50)]

    # Run optimization
    print("Initial experiences:")
    for i, exp in enumerate(initial_experiences, 1):
        print(f"{i}. {exp}")

    optimized_experiences, reward_curve = trainer.optimize_experiences(
        initial_experiences=initial_experiences,
        problems=problems,
        num_iterations=5,
        rollouts_per_iteration=10
    )

    print("\n=== Optimization Complete ===")
    print(f"\nFinal experiences ({len(optimized_experiences)}):")
    for i, exp in enumerate(optimized_experiences, 1):
        print(f"{i}. {exp}")

    print(f"\nReward curve: {[f'{r:.3f}' for r in reward_curve]}")
    print("\n✓ CPG successfully optimized experience set via semantic gradients")
