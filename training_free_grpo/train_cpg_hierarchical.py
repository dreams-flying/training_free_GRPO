"""
Training-Free GRPO with Context-Policy Gradient and Hierarchical Retrieval

This script integrates two major innovations:

1. Context-Policy Gradient (CPG):
   - Treats experience updates as gradient descent in semantic space
   - LLM acts as implicit gradient estimator
   - Optimizes experiences based on reward signals

2. Hierarchical Retrieval-Augmented Prior:
   - Organizes experiences in 3-level hierarchy (meta → domain → task)
   - Dynamically retrieves relevant experiences per problem
   - Enables cross-domain transfer and scaling

Together, these create a self-improving, scalable training-free RL system.
"""

import argparse
import json
from pathlib import Path
from typing import List, Dict, Tuple
import numpy as np

from training_free_grpo.context_policy_gradient import (
    ContextPolicyGradient,
    ExperienceUpdate
)
from training_free_grpo.hierarchical_retrieval import (
    HierarchicalExperienceLibrary,
    ProblemClassifier,
    Experience
)
from training_free_grpo.llm import LLM


class CPGHierarchicalTrainer:
    """
    Integrated trainer combining CPG and Hierarchical Retrieval.

    Training Loop:
    1. For each problem:
        a. Classify problem (domain, task_type, difficulty)
        b. Retrieve relevant experiences from hierarchy
        c. Run rollout with retrieved experiences
        d. Record effectiveness of used experiences

    2. After batch of problems:
        a. Compute semantic gradients via CPG
        b. Update experience library
        c. Reorganize hierarchy if needed
    """

    def __init__(
        self,
        llm_client: LLM,
        experience_library: HierarchicalExperienceLibrary,
        cpg_learning_rate: float = 0.3,
        update_frequency: int = 20
    ):
        self.llm = llm_client
        self.library = experience_library
        self.classifier = ProblemClassifier(llm_client)

        # CPG optimizer for experience updates
        self.cpg = ContextPolicyGradient(
            llm_client=llm_client,
            learning_rate=cpg_learning_rate
        )

        self.update_frequency = update_frequency

        # Tracking
        self.reward_history = []
        self.rollout_count = 0
        self.experience_usage = {}  # Track which experiences were used

    def train(
        self,
        problems: List[str],
        num_rollouts_per_problem: int = 4,
        batch_size: int = 20
    ):
        """
        Main training loop.

        Args:
            problems: List of problems to solve
            num_rollouts_per_problem: Rollouts per problem for GRPO
            batch_size: Problems per CPG update
        """
        print("="*70)
        print("Training-Free GRPO with CPG + Hierarchical Retrieval")
        print("="*70)

        reward_trajectory = []
        all_rollouts = []

        for problem_idx, problem in enumerate(problems):
            print(f"\n[{problem_idx + 1}/{len(problems)}] Processing problem...")
            print(f"Problem: {problem[:80]}...")

            # === Step 1: Classify problem ===
            domain, task_type, difficulty = self.classifier.classify(problem)
            print(f"  Classified: domain={domain}, task={task_type}, difficulty={difficulty}")

            # === Step 2: Retrieve relevant experiences ===
            relevant_experiences = self.library.retrieve_by_difficulty(
                problem=problem,
                difficulty=difficulty,
                domain=domain
            )

            print(f"  Retrieved {len(relevant_experiences)} relevant experiences:")
            for i, exp in enumerate(relevant_experiences[:3], 1):
                print(f"    {i}. [{exp.level}] {exp.content[:60]}...")

            # === Step 3: Run GRPO rollouts ===
            rollouts = self._run_grpo_rollouts(
                problem=problem,
                experiences=relevant_experiences,
                num_rollouts=num_rollouts_per_problem
            )

            all_rollouts.extend(rollouts)

            # === Step 4: Update experience effectiveness ===
            best_reward = max(r['reward'] for r in rollouts)
            avg_reward = np.mean([r['reward'] for r in rollouts])

            print(f"  Rewards: best={best_reward:.3f}, avg={avg_reward:.3f}")

            # Mark experiences as successful if avg reward is high
            success = avg_reward > 0.5
            for exp in relevant_experiences:
                self.library.update_effectiveness(exp, success=success)
                self.experience_usage[exp.content] = self.experience_usage.get(exp.content, 0) + 1

            reward_trajectory.append((problem, avg_reward))
            self.reward_history.append(avg_reward)
            self.rollout_count += len(rollouts)

            # === Step 5: Periodic CPG update ===
            if (problem_idx + 1) % self.update_frequency == 0:
                print("\n" + "="*70)
                print(f"CPG Update (after {problem_idx + 1} problems)")
                print("="*70)
                self._update_experiences_via_cpg(
                    reward_trajectory=reward_trajectory[-self.update_frequency:],
                    domain=domain
                )

        # Final statistics
        self._print_final_statistics()

    def _run_grpo_rollouts(
        self,
        problem: str,
        experiences: List[Experience],
        num_rollouts: int
    ) -> List[Dict]:
        """
        Run GRPO rollouts for a problem with given experiences.

        Args:
            problem: Problem to solve
            experiences: Retrieved experiences to use as context
            num_rollouts: Number of rollouts

        Returns:
            List of rollout results
        """
        # Build prompt with experiences as context
        experience_text = "\n".join([
            f"- {exp.content}" for exp in experiences
        ])

        prompt = f"""You are solving the following problem. Use the experiences below as guidance.

**Experiences:**
{experience_text}

**Problem:**
{problem}

**Your solution:**"""

        rollouts = []

        for i in range(num_rollouts):
            # Generate response
            response = self.llm.generate(prompt, temperature=0.7)

            # Simulate reward (in practice, use actual verification)
            reward = self._evaluate_response(problem, response)

            rollouts.append({
                'problem': problem,
                'response': response,
                'reward': reward,
                'experiences_used': [exp.content for exp in experiences]
            })

        return rollouts

    def _evaluate_response(self, problem: str, response: str) -> float:
        """
        Evaluate response quality.

        In practice, this would use actual verification (e.g., test cases, correct answer).
        Here we use a simplified simulation.
        """
        # Simplified: longer, more structured responses get higher rewards
        score = 0.0

        # Length factor
        if len(response) > 100:
            score += 0.3

        # Structure factor (has steps)
        if any(marker in response.lower() for marker in ["step", "first", "then", "finally"]):
            score += 0.3

        # Verification factor
        if any(marker in response.lower() for marker in ["verify", "check", "confirm"]):
            score += 0.2

        # Add randomness to simulate actual variance
        score += np.random.uniform(-0.2, 0.2)

        return max(0.0, min(1.0, score))

    def _update_experiences_via_cpg(
        self,
        reward_trajectory: List[Tuple[str, float]],
        domain: str
    ):
        """
        Update experience library using Context-Policy Gradient.

        Args:
            reward_trajectory: Recent (problem, reward) pairs
            domain: Current problem domain
        """
        # Get current experiences from library
        current_experiences = self._get_all_experiences_as_text()

        # Compute semantic gradients
        print("  Computing semantic gradients...")
        gradients = self.cpg.compute_semantic_gradient(
            experiences=current_experiences,
            reward_trajectory=reward_trajectory,
            problem_context=domain
        )

        print(f"  Generated {len(gradients)} gradient updates:")
        for i, grad in enumerate(gradients[:5], 1):
            print(f"    {i}. {grad.operation:10s} [mag={grad.gradient_magnitude:.2f}]: {grad.reasoning[:60]}...")

        # Apply gradients to create new experiences
        print("  Applying gradients to experience library...")
        self._apply_gradients_to_library(gradients, domain)

        # Show updated statistics
        stats = self.library.get_statistics()
        print(f"  Updated library: {stats['total_experiences']} total experiences")
        print(f"  Avg effectiveness: {stats['avg_effectiveness']:.3f}")

    def _get_all_experiences_as_text(self) -> List[str]:
        """Get all experiences as text list for CPG."""
        return [exp.content for exp in self.library.all_experiences]

    def _apply_gradients_to_library(
        self,
        gradients: List[ExperienceUpdate],
        domain: str
    ):
        """
        Apply CPG gradients to update hierarchical library.

        This translates gradient updates into hierarchical experience additions/modifications.
        """
        for grad in gradients:
            if grad.gradient_magnitude < 0.15:  # Skip weak gradients
                continue

            if grad.operation == "add" and grad.new_content:
                # Classify new experience into hierarchy
                level = self._classify_experience_level(grad.new_content)

                self.library.add_experience(
                    content=grad.new_content,
                    level=level,
                    domain=domain if level != "meta" else None
                )

            elif grad.operation == "modify":
                # Find and modify experience
                for exp in self.library.all_experiences:
                    if self._text_similarity(exp.content, grad.target_experience) > 0.7:
                        # Create new version
                        self.library.add_experience(
                            content=grad.new_content,
                            level=exp.level,
                            domain=exp.domain,
                            task_type=exp.task_type
                        )
                        break

            elif grad.operation == "strengthen":
                # Boost effectiveness of matching experiences
                for exp in self.library.all_experiences:
                    if self._text_similarity(exp.content, grad.target_experience) > 0.7:
                        exp.effectiveness_score = min(1.0, exp.effectiveness_score + 0.2)

    def _classify_experience_level(self, experience: str) -> str:
        """
        Classify experience into hierarchy level.

        Meta: Domain-agnostic strategies
        Domain: Domain-specific but task-agnostic
        Task: Task-specific tactics
        """
        exp_lower = experience.lower()

        # Meta-level indicators (universal strategies)
        meta_keywords = ["always", "when solving", "general", "any problem", "universal"]
        if any(kw in exp_lower for kw in meta_keywords):
            return "meta"

        # Task-level indicators (specific tactics)
        task_keywords = ["quadratic", "linear", "specific", "particular case", "when you see"]
        if any(kw in exp_lower for kw in task_keywords):
            return "task"

        # Default to domain level
        return "domain"

    def _text_similarity(self, text1: str, text2: str) -> float:
        """Simple text similarity."""
        words1 = set(text1.lower().split())
        words2 = set(text2.lower().split())

        if not words1 or not words2:
            return 0.0

        intersection = words1.intersection(words2)
        union = words1.union(words2)

        return len(intersection) / len(union) if union else 0.0

    def _print_final_statistics(self):
        """Print final training statistics."""
        print("\n" + "="*70)
        print("Training Complete - Final Statistics")
        print("="*70)

        print(f"\nRollouts: {self.rollout_count}")
        print(f"Avg Reward: {np.mean(self.reward_history):.3f}")
        print(f"Reward Std: {np.std(self.reward_history):.3f}")

        # Reward trend
        if len(self.reward_history) > 10:
            early = np.mean(self.reward_history[:10])
            late = np.mean(self.reward_history[-10:])
            improvement = late - early
            print(f"Improvement: {improvement:+.3f} (early={early:.3f}, late={late:.3f})")

        # Library statistics
        stats = self.library.get_statistics()
        print(f"\nExperience Library:")
        print(f"  Total: {stats['total_experiences']}")
        print(f"  Meta: {stats['meta_experiences']}")
        print(f"  Domains: {list(stats['domain_breakdown'].keys())}")
        print(f"  Avg Effectiveness: {stats['avg_effectiveness']:.3f}")

        # Most used experiences
        print(f"\nTop 5 Most Used Experiences:")
        sorted_usage = sorted(self.experience_usage.items(), key=lambda x: x[1], reverse=True)
        for i, (exp, count) in enumerate(sorted_usage[:5], 1):
            print(f"  {i}. [{count:3d}x] {exp[:60]}...")

        # Most effective experiences
        print(f"\nTop 5 Most Effective Experiences:")
        for i, exp in enumerate(stats['most_effective'][:5], 1):
            print(f"  {i}. [{exp.effectiveness_score:.3f}] {exp.content[:60]}...")


def main():
    parser = argparse.ArgumentParser(
        description="Training-Free GRPO with CPG and Hierarchical Retrieval"
    )
    parser.add_argument("--dataset", type=str, default="AIME24", help="Dataset name")
    parser.add_argument("--num_problems", type=int, default=50, help="Number of problems")
    parser.add_argument("--cpg_learning_rate", type=float, default=0.3, help="CPG learning rate")
    parser.add_argument("--update_frequency", type=int, default=20, help="CPG update frequency")
    parser.add_argument("--library_path", type=str, help="Path to existing experience library JSON")
    parser.add_argument("--save_library", type=str, help="Path to save final library")

    args = parser.parse_args()

    # Initialize LLM
    print("Initializing LLM client...")
    llm = LLM()

    # Initialize or load experience library
    library = HierarchicalExperienceLibrary()

    if args.library_path:
        print(f"Loading experience library from {args.library_path}...")
        library.import_from_json(args.library_path)
    else:
        print("Initializing new experience library with seed experiences...")
        _initialize_seed_experiences(library)

    # Create trainer
    trainer = CPGHierarchicalTrainer(
        llm_client=llm,
        experience_library=library,
        cpg_learning_rate=args.cpg_learning_rate,
        update_frequency=args.update_frequency
    )

    # Load problems (placeholder - replace with actual dataset loading)
    print(f"Loading {args.dataset} dataset...")
    problems = _load_problems(args.dataset, args.num_problems)

    # Train
    trainer.train(
        problems=problems,
        num_rollouts_per_problem=4,
        batch_size=args.update_frequency
    )

    # Save library
    if args.save_library:
        print(f"\nSaving experience library to {args.save_library}...")
        library.export_to_json(args.save_library)
        print("✓ Saved")


def _initialize_seed_experiences(library: HierarchicalExperienceLibrary):
    """Initialize library with seed experiences."""
    # Meta-level
    library.add_experience(
        "Break complex problems into smaller, manageable steps",
        level="meta",
        tags={"decomposition"}
    )
    library.add_experience(
        "Verify each intermediate result before proceeding",
        level="meta",
        tags={"verification"}
    )

    # Math domain
    library.add_experience(
        "Simplify equations using algebraic manipulation",
        level="domain",
        domain="math",
        tags={"algebra"}
    )

    print(f"  Initialized with {library.get_statistics()['total_experiences']} seed experiences")


def _load_problems(dataset: str, num_problems: int) -> List[str]:
    """Load problems from dataset (placeholder)."""
    # In practice, load from actual dataset
    return [
        f"Sample problem {i+1} from {dataset}"
        for i in range(num_problems)
    ]


if __name__ == "__main__":
    main()
