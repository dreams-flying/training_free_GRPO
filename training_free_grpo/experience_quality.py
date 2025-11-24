"""
Experience Quality Assessment Module

Evaluates and filters experiences based on multiple quality metrics:
1. Generality: How broadly applicable is the experience?
2. Actionability: Does it provide concrete guidance?
3. Distinctiveness: Is it different from existing experiences?
4. Empirical Effectiveness: Does it improve performance?
"""

import json
import re
from typing import List, Dict, Tuple
from collections import defaultdict
import numpy as np
from training_free_grpo.llm import LLM


class ExperienceQualityAssessor:
    """Assesses and filters experiences based on quality metrics."""

    def __init__(self, quality_threshold: float = 0.6):
        self.llm = LLM()
        self.quality_threshold = quality_threshold

    def assess_experience_quality(self, experience: str, domain: str = "general") -> Dict[str, float]:
        """
        Assess experience quality across multiple dimensions.

        Returns:
            Dict with scores: generality, actionability, distinctiveness, overall
        """
        prompt = f"""
Evaluate the following experience for a {domain} problem-solving agent:

EXPERIENCE: {experience}

Rate the experience on these dimensions (0.0-1.0):

1. GENERALITY: Can it apply to many problems or just specific cases?
   - 1.0: Highly general, applies to broad categories
   - 0.5: Moderately general, applies to some subcategories
   - 0.0: Too specific, only applies to one problem type

2. ACTIONABILITY: Does it provide clear, actionable guidance?
   - 1.0: Specific steps or strategies clearly described
   - 0.5: General direction but lacks specifics
   - 0.0: Vague or unclear guidance

3. CLARITY: Is it concise and easy to understand?
   - 1.0: Clear, concise, no ambiguity
   - 0.5: Understandable but could be clearer
   - 0.0: Confusing or overly complex

Respond in JSON format:
```json
{{
  "generality": 0.8,
  "actionability": 0.7,
  "clarity": 0.9,
  "reasoning": "Brief explanation"
}}
```
"""

        try:
            response = self.llm.chat(prompt, temperature=0.0, max_tokens=512)
            response = response.split("```json")[-1].split("```")[0].strip()
            scores = json.loads(response)

            # Calculate overall score as weighted average
            overall = (
                0.4 * scores.get("generality", 0.5) +
                0.35 * scores.get("actionability", 0.5) +
                0.25 * scores.get("clarity", 0.5)
            )
            scores["overall"] = overall

            return scores
        except Exception as e:
            print(f"Warning: Quality assessment failed: {e}")
            # Return default moderate scores
            return {
                "generality": 0.5,
                "actionability": 0.5,
                "clarity": 0.5,
                "overall": 0.5,
                "reasoning": "Assessment failed"
            }

    def calculate_distinctiveness(
        self,
        new_experience: str,
        existing_experiences: List[str]
    ) -> float:
        """
        Calculate how different a new experience is from existing ones.
        Uses simple token overlap as proxy (could use embeddings for better results).

        Returns:
            Float in [0, 1]: 1.0 = completely distinct, 0.0 = duplicate
        """
        if not existing_experiences:
            return 1.0

        def tokenize(text: str) -> set:
            """Simple word tokenization."""
            return set(re.findall(r'\w+', text.lower()))

        new_tokens = tokenize(new_experience)
        if not new_tokens:
            return 0.0

        # Calculate max overlap with any existing experience
        max_overlap = 0.0
        for exp in existing_experiences:
            exp_tokens = tokenize(exp)
            if not exp_tokens:
                continue

            # Jaccard similarity
            intersection = len(new_tokens & exp_tokens)
            union = len(new_tokens | exp_tokens)
            similarity = intersection / union if union > 0 else 0.0
            max_overlap = max(max_overlap, similarity)

        # Distinctiveness is 1 - max_similarity
        return 1.0 - max_overlap

    def filter_experiences(
        self,
        experiences: Dict[str, str],
        domain: str = "general",
        min_quality: float = None
    ) -> Dict[str, Tuple[str, float]]:
        """
        Filter experiences based on quality threshold.

        Args:
            experiences: Dict of {id: experience_text}
            domain: Problem domain (math/web/general)
            min_quality: Minimum overall quality score (uses self.quality_threshold if None)

        Returns:
            Dict of {id: (experience_text, quality_score)} for experiences above threshold
        """
        min_quality = min_quality or self.quality_threshold

        filtered = {}
        all_exp_texts = list(experiences.values())

        for exp_id, exp_text in experiences.items():
            # Assess quality
            scores = self.assess_experience_quality(exp_text, domain)

            # Calculate distinctiveness
            other_exps = [e for i, e in enumerate(all_exp_texts) if all_exp_texts[i] != exp_text]
            distinctiveness = self.calculate_distinctiveness(exp_text, other_exps)

            # Combined score: quality + distinctiveness bonus
            combined_score = scores["overall"] * 0.8 + distinctiveness * 0.2

            if combined_score >= min_quality:
                filtered[exp_id] = (exp_text, combined_score)
                print(f"[Quality Filter] {exp_id}: quality={scores['overall']:.2f}, "
                      f"distinct={distinctiveness:.2f}, combined={combined_score:.2f} ✓")
            else:
                print(f"[Quality Filter] {exp_id}: combined={combined_score:.2f} ✗ (below {min_quality})")

        return filtered

    def rank_experiences(
        self,
        experiences: Dict[str, str],
        domain: str = "general"
    ) -> List[Tuple[str, str, float]]:
        """
        Rank experiences by quality score.

        Returns:
            List of (id, experience, score) tuples sorted by score descending
        """
        filtered = self.filter_experiences(experiences, domain, min_quality=0.0)
        ranked = [(id, exp, score) for id, (exp, score) in filtered.items()]
        ranked.sort(key=lambda x: x[2], reverse=True)
        return ranked


class EmpiricalEffectivenessTracker:
    """Tracks how effective each experience is in practice."""

    def __init__(self, save_path: str = "experience_effectiveness.json"):
        self.save_path = save_path
        self.effectiveness_data = defaultdict(lambda: {"usage": 0, "success": 0, "avg_reward": 0.0})
        self.load()

    def load(self):
        """Load effectiveness data from file."""
        try:
            with open(self.save_path, 'r') as f:
                data = json.load(f)
                self.effectiveness_data.update(data)
                print(f"[Effectiveness Tracker] Loaded data for {len(data)} experiences")
        except FileNotFoundError:
            print(f"[Effectiveness Tracker] No existing data found, starting fresh")

    def save(self):
        """Save effectiveness data to file."""
        with open(self.save_path, 'w') as f:
            json.dump(dict(self.effectiveness_data), f, indent=2)

    def record_usage(self, experience_ids: List[str], reward: float):
        """
        Record that certain experiences were used and resulted in a reward.

        Args:
            experience_ids: List of experience IDs that were in the prompt
            reward: Resulting reward (0.0 or 1.0)
        """
        for exp_id in experience_ids:
            data = self.effectiveness_data[exp_id]
            data["usage"] += 1
            if reward > 0.5:
                data["success"] += 1

            # Update running average
            old_avg = data["avg_reward"]
            n = data["usage"]
            data["avg_reward"] = (old_avg * (n - 1) + reward) / n

    def get_effectiveness(self, experience_id: str) -> Dict[str, float]:
        """
        Get effectiveness metrics for an experience.

        Returns:
            Dict with: usage_count, success_rate, avg_reward
        """
        data = self.effectiveness_data[experience_id]
        success_rate = data["success"] / data["usage"] if data["usage"] > 0 else 0.0
        return {
            "usage_count": data["usage"],
            "success_rate": success_rate,
            "avg_reward": data["avg_reward"]
        }

    def get_top_experiences(self, n: int = 10) -> List[Tuple[str, float]]:
        """
        Get top N experiences by success rate (with minimum usage threshold).

        Returns:
            List of (experience_id, success_rate) tuples
        """
        min_usage = 5  # Require at least 5 usages for reliability

        candidates = [
            (exp_id, data["success"] / data["usage"])
            for exp_id, data in self.effectiveness_data.items()
            if data["usage"] >= min_usage
        ]

        candidates.sort(key=lambda x: x[1], reverse=True)
        return candidates[:n]

    def prune_ineffective_experiences(
        self,
        experiences: Dict[str, str],
        min_success_rate: float = 0.3,
        min_usage: int = 10
    ) -> Dict[str, str]:
        """
        Remove experiences that have low empirical effectiveness.

        Args:
            experiences: Dict of {id: experience_text}
            min_success_rate: Minimum success rate to keep
            min_usage: Minimum usage count before pruning (avoid premature pruning)

        Returns:
            Filtered dict of experiences
        """
        pruned = {}

        for exp_id, exp_text in experiences.items():
            if exp_id not in self.effectiveness_data:
                # Keep new experiences that haven't been tested yet
                pruned[exp_id] = exp_text
                continue

            data = self.effectiveness_data[exp_id]
            if data["usage"] < min_usage:
                # Not enough data, keep it
                pruned[exp_id] = exp_text
                continue

            success_rate = data["success"] / data["usage"]
            if success_rate >= min_success_rate:
                pruned[exp_id] = exp_text
                print(f"[Effectiveness] Keep {exp_id}: {success_rate:.2%} success over {data['usage']} uses")
            else:
                print(f"[Effectiveness] Prune {exp_id}: {success_rate:.2%} success over {data['usage']} uses ✗")

        return pruned


if __name__ == "__main__":
    # Example usage
    assessor = ExperienceQualityAssessor(quality_threshold=0.6)

    # Test experiences
    test_experiences = {
        "G0": "When solving a problem, break it down into smaller subproblems.",
        "G1": "For problem 123 on page 45, use formula x = y + z.",
        "G2": "Always verify your final answer against the problem constraints.",
        "G3": "Use tools systematically: first search, then analyze, finally synthesize.",
    }

    print("=== Quality Assessment ===")
    ranked = assessor.rank_experiences(test_experiences, domain="math")

    print("\n=== Ranked Experiences ===")
    for id, exp, score in ranked:
        print(f"{id} ({score:.2f}): {exp}")

    print("\n=== Filtered Experiences (threshold=0.6) ===")
    filtered = assessor.filter_experiences(test_experiences, domain="math", min_quality=0.6)
    for id, (exp, score) in filtered.items():
        print(f"{id} ({score:.2f}): {exp}")

    # Test effectiveness tracker
    print("\n=== Effectiveness Tracking ===")
    tracker = EmpiricalEffectivenessTracker(save_path="test_effectiveness.json")

    # Simulate some usage
    tracker.record_usage(["G0", "G2"], reward=1.0)
    tracker.record_usage(["G0", "G3"], reward=1.0)
    tracker.record_usage(["G1"], reward=0.0)
    tracker.record_usage(["G1"], reward=0.0)
    tracker.record_usage(["G1"], reward=0.0)
    tracker.save()

    print("\nTop experiences:")
    for exp_id, success_rate in tracker.get_top_experiences(3):
        print(f"  {exp_id}: {success_rate:.2%} success rate")
