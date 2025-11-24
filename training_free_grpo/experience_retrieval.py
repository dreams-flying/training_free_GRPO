"""
Dynamic Experience Retrieval Module

Selects most relevant experiences for each problem based on:
1. Semantic similarity
2. Problem difficulty
3. Historical effectiveness
4. Diversity
"""

import json
import re
from typing import List, Dict, Tuple
import numpy as np
from training_free_grpo.llm import LLM


class ExperienceRetriever:
    """
    Retrieves most relevant experiences for a given problem.

    Uses multiple strategies:
    - Semantic similarity (via LLM-based matching)
    - Difficulty-based filtering
    - Effectiveness-based ranking
    - Diversity promotion
    """

    def __init__(
        self,
        top_k: int = 5,
        use_semantic_sim: bool = True,
        use_effectiveness: bool = True,
        diversity_penalty: float = 0.2
    ):
        """
        Args:
            top_k: Number of experiences to retrieve
            use_semantic_sim: Whether to use semantic similarity
            use_effectiveness: Whether to weight by effectiveness
            diversity_penalty: Penalty for similar experiences (0-1)
        """
        self.llm = LLM()
        self.top_k = top_k
        self.use_semantic_sim = use_semantic_sim
        self.use_effectiveness = use_effectiveness
        self.diversity_penalty = diversity_penalty

    def compute_semantic_similarity(
        self,
        problem: str,
        experience: str
    ) -> float:
        """
        Compute how relevant an experience is to a problem using LLM.

        Returns:
            Float in [0, 1]: relevance score
        """
        prompt = f"""
Rate how relevant the following EXPERIENCE is to solving the given PROBLEM.

PROBLEM: {problem}

EXPERIENCE: {experience}

Consider:
1. Does the experience provide strategies applicable to this problem?
2. Does it address similar challenges or patterns?
3. Would it help guide the problem-solving process?

Rate relevance on a scale of 0-10, where:
- 10: Highly relevant, directly applicable
- 5: Somewhat relevant, general guidance
- 0: Not relevant at all

Respond with ONLY a number from 0-10.
"""

        try:
            response = self.llm.chat(prompt, temperature=0.0, max_tokens=10)
            # Extract number
            match = re.search(r'\d+', response)
            if match:
                score = int(match.group())
                return min(max(score, 0), 10) / 10.0
            return 0.5  # Default if parsing fails
        except Exception as e:
            print(f"Warning: Similarity computation failed: {e}")
            return 0.5

    def compute_keyword_similarity(
        self,
        problem: str,
        experience: str
    ) -> float:
        """
        Fast keyword-based similarity as fallback.

        Returns:
            Float in [0, 1]: Jaccard similarity
        """
        def tokenize(text: str) -> set:
            # Remove common words and tokenize
            stopwords = {'the', 'a', 'an', 'is', 'are', 'was', 'were', 'to', 'of', 'in', 'for', 'on', 'with'}
            words = set(re.findall(r'\w+', text.lower()))
            return words - stopwords

        problem_words = tokenize(problem)
        exp_words = tokenize(experience)

        if not problem_words or not exp_words:
            return 0.0

        intersection = len(problem_words & exp_words)
        union = len(problem_words | exp_words)
        return intersection / union if union > 0 else 0.0

    def estimate_problem_difficulty(self, problem: str) -> str:
        """
        Estimate problem difficulty: easy, medium, or hard.

        Based on:
        - Length and complexity
        - Keywords indicating difficulty
        - Structural patterns
        """
        # Simple heuristics (could use LLM for better estimates)
        length = len(problem.split())

        # Difficulty indicators
        hard_keywords = ['prove', 'derive', 'optimize', 'maximum', 'minimum', 'complex', 'advanced']
        medium_keywords = ['calculate', 'find', 'determine', 'solve', 'compute']

        problem_lower = problem.lower()
        hard_count = sum(1 for kw in hard_keywords if kw in problem_lower)
        medium_count = sum(1 for kw in medium_keywords if kw in problem_lower)

        if hard_count >= 2 or length > 150:
            return "hard"
        elif medium_count >= 1 or length > 80:
            return "medium"
        else:
            return "easy"

    def retrieve_experiences(
        self,
        problem: str,
        experiences: Dict[str, str],
        effectiveness_scores: Dict[str, float] = None,
        difficulty_filter: str = None
    ) -> List[Tuple[str, str, float]]:
        """
        Retrieve top-k most relevant experiences for a problem.

        Args:
            problem: The problem text
            experiences: Dict of {id: experience_text}
            effectiveness_scores: Optional dict of {id: effectiveness_score}
            difficulty_filter: Optional filter by difficulty (easy/medium/hard)

        Returns:
            List of (id, experience, relevance_score) tuples, sorted by relevance
        """
        if not experiences:
            return []

        # Estimate problem difficulty
        problem_difficulty = self.estimate_problem_difficulty(problem)
        print(f"[Retrieval] Problem difficulty: {problem_difficulty}")

        # Compute relevance scores
        scored_experiences = []

        for exp_id, exp_text in experiences.items():
            # Base score: semantic or keyword similarity
            if self.use_semantic_sim and len(experiences) <= 20:
                # Use LLM for small experience sets
                sim_score = self.compute_semantic_similarity(problem, exp_text)
            else:
                # Use fast keyword similarity for large sets
                sim_score = self.compute_keyword_similarity(problem, exp_text)

            # Boost by effectiveness if available
            if self.use_effectiveness and effectiveness_scores and exp_id in effectiveness_scores:
                effectiveness = effectiveness_scores[exp_id]
                # Weighted combination: 70% similarity, 30% effectiveness
                final_score = 0.7 * sim_score + 0.3 * effectiveness
            else:
                final_score = sim_score

            scored_experiences.append((exp_id, exp_text, final_score))

        # Sort by score descending
        scored_experiences.sort(key=lambda x: x[2], reverse=True)

        # Apply diversity penalty to avoid redundant experiences
        if self.diversity_penalty > 0:
            scored_experiences = self._promote_diversity(scored_experiences)

        # Return top-k
        top_k = scored_experiences[:self.top_k]

        print(f"[Retrieval] Selected {len(top_k)} experiences:")
        for exp_id, _, score in top_k:
            print(f"  {exp_id}: relevance={score:.3f}")

        return top_k

    def _promote_diversity(
        self,
        scored_experiences: List[Tuple[str, str, float]]
    ) -> List[Tuple[str, str, float]]:
        """
        Re-rank experiences to promote diversity using Maximal Marginal Relevance.

        Balances relevance and diversity.
        """
        if len(scored_experiences) <= 1:
            return scored_experiences

        selected = []
        remaining = scored_experiences.copy()

        # Always take the top one
        selected.append(remaining.pop(0))

        while remaining and len(selected) < len(scored_experiences):
            # For each remaining, compute diversity penalty
            best_idx = 0
            best_score = -float('inf')

            for idx, (exp_id, exp_text, rel_score) in enumerate(remaining):
                # Compute max similarity with already selected
                max_sim = 0.0
                for _, selected_text, _ in selected:
                    sim = self.compute_keyword_similarity(exp_text, selected_text)
                    max_sim = max(max_sim, sim)

                # MMR score: relevance - diversity_penalty * max_similarity
                mmr_score = rel_score - self.diversity_penalty * max_sim

                if mmr_score > best_score:
                    best_score = mmr_score
                    best_idx = idx

            # Move best to selected
            selected.append(remaining.pop(best_idx))

        return selected

    def format_retrieved_experiences(
        self,
        retrieved: List[Tuple[str, str, float]]
    ) -> str:
        """
        Format retrieved experiences for insertion into prompt.

        Returns:
            Formatted string ready for PROBLEM_WITH_EXPERIENCE_TEMPLATE
        """
        if not retrieved:
            return "None"

        formatted_lines = []
        for idx, (exp_id, exp_text, score) in enumerate(retrieved):
            # Format: [0]. Experience text (relevance: 0.85)
            formatted_lines.append(f"[{idx}]. {exp_text}")

        return "\n".join(formatted_lines)


class DifficultyAdaptiveRetriever(ExperienceRetriever):
    """
    Advanced retriever that adapts strategy based on problem difficulty.

    - Easy problems: Use fewer, more general experiences
    - Hard problems: Use more, more specific experiences
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.difficulty_to_k = {
            "easy": 3,
            "medium": 5,
            "hard": 7
        }

    def retrieve_experiences(
        self,
        problem: str,
        experiences: Dict[str, str],
        effectiveness_scores: Dict[str, float] = None,
        difficulty_filter: str = None
    ) -> List[Tuple[str, str, float]]:
        """
        Override to adapt top_k based on problem difficulty.
        """
        # Estimate difficulty
        difficulty = self.estimate_problem_difficulty(problem)

        # Adjust top_k
        original_k = self.top_k
        self.top_k = self.difficulty_to_k.get(difficulty, self.top_k)

        print(f"[Adaptive Retrieval] Difficulty={difficulty}, using top_k={self.top_k}")

        # Call parent's retrieve
        result = super().retrieve_experiences(
            problem, experiences, effectiveness_scores, difficulty_filter
        )

        # Restore original k
        self.top_k = original_k

        return result


if __name__ == "__main__":
    # Example usage
    retriever = DifficultyAdaptiveRetriever(
        top_k=5,
        use_semantic_sim=False,  # Use fast keyword similarity for demo
        diversity_penalty=0.3
    )

    # Test problem
    problem = """
    Find the maximum value of the function f(x) = -x^2 + 4x + 5
    on the interval [-2, 3]. Justify your answer.
    """

    # Test experiences
    experiences = {
        "G0": "When solving optimization problems, find critical points by taking derivatives.",
        "G1": "Always check endpoints when optimizing over a closed interval.",
        "G2": "For quadratic functions, the vertex gives the maximum or minimum.",
        "G3": "When dealing with web queries, use specific search terms.",
        "G4": "Break complex problems into smaller subproblems.",
        "G5": "Verify your answer by substituting back into the original equation.",
        "G6": "For calculus problems, remember to check the second derivative.",
    }

    # Mock effectiveness scores
    effectiveness = {
        "G0": 0.85,
        "G1": 0.78,
        "G2": 0.90,
        "G3": 0.45,
        "G4": 0.70,
        "G5": 0.65,
        "G6": 0.82,
    }

    print("=== Problem ===")
    print(problem)
    print()

    print("=== Retrieved Experiences ===")
    retrieved = retriever.retrieve_experiences(
        problem,
        experiences,
        effectiveness_scores=effectiveness
    )

    print("\n=== Formatted for Prompt ===")
    formatted = retriever.format_retrieved_experiences(retrieved)
    print(formatted)
