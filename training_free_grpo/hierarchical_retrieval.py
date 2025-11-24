"""
Hierarchical Retrieval-Augmented Prior for Training-Free GRPO

This module implements a hierarchical experience organization with semantic retrieval,
allowing dynamic selection of the most relevant experiences for each problem.

Key Innovation:
    Instead of using all experiences for all problems, we:
    1. Organize experiences into a 3-level hierarchy (meta → domain → task)
    2. Use semantic similarity for retrieval
    3. Dynamically select top-k relevant experiences per problem
    4. Support cross-domain transfer via meta-experiences

Benefits:
    - Reduces context length (only relevant experiences)
    - Improves generalization (hierarchical organization)
    - Enables transfer learning across domains
    - Scalable to large experience libraries
"""

from typing import List, Dict, Tuple, Optional, Set
import numpy as np
from dataclasses import dataclass, field
from collections import defaultdict
import json


@dataclass
class Experience:
    """Represents a single experience with metadata."""
    content: str
    level: str  # "meta", "domain", "task"
    domain: Optional[str] = None  # e.g., "math", "code", "web"
    task_type: Optional[str] = None  # e.g., "algebra", "geometry"
    embedding: Optional[np.ndarray] = None
    effectiveness_score: float = 0.0
    usage_count: int = 0
    success_rate: float = 0.0
    tags: Set[str] = field(default_factory=set)


class HierarchicalExperienceLibrary:
    """
    Organizes experiences into a 3-level hierarchy:

    Level 1 (Meta): Domain-agnostic high-level strategies
        - "Break complex problems into smaller steps"
        - "Verify intermediate results before proceeding"

    Level 2 (Domain): Domain-specific but task-agnostic strategies
        - Math: "Use algebraic manipulation to simplify equations"
        - Code: "Write test cases before implementation"

    Level 3 (Task): Task-specific tactics
        - Algebra: "When solving quadratic equations, check for factorization first"
        - Web Search: "Use site: operator to search within specific domains"
    """

    def __init__(self, embedding_function=None):
        """
        Args:
            embedding_function: Function to convert text → vector (e.g., sentence-transformers)
        """
        self.embedding_fn = embedding_function or self._simple_embedding

        # Hierarchical storage
        self.meta_experiences: List[Experience] = []
        self.domain_experiences: Dict[str, List[Experience]] = defaultdict(list)
        self.task_experiences: Dict[Tuple[str, str], List[Experience]] = defaultdict(list)

        # Flat index for fast retrieval
        self.all_experiences: List[Experience] = []
        self.experience_embeddings: Optional[np.ndarray] = None

    def add_experience(
        self,
        content: str,
        level: str,
        domain: Optional[str] = None,
        task_type: Optional[str] = None,
        tags: Optional[Set[str]] = None
    ) -> Experience:
        """Add a new experience to the hierarchy."""
        exp = Experience(
            content=content,
            level=level,
            domain=domain,
            task_type=task_type,
            tags=tags or set()
        )

        # Compute embedding
        exp.embedding = self.embedding_fn(content)

        # Add to appropriate level
        if level == "meta":
            self.meta_experiences.append(exp)
        elif level == "domain" and domain:
            self.domain_experiences[domain].append(exp)
        elif level == "task" and domain and task_type:
            self.task_experiences[(domain, task_type)].append(exp)

        # Add to flat index
        self.all_experiences.append(exp)
        self._rebuild_index()

        return exp

    def retrieve_experiences(
        self,
        problem: str,
        domain: Optional[str] = None,
        task_type: Optional[str] = None,
        top_k: int = 5,
        include_meta: bool = True,
        diversity_penalty: float = 0.3
    ) -> List[Experience]:
        """
        Retrieve the most relevant experiences for a problem.

        Args:
            problem: The problem text
            domain: Problem domain (if known)
            task_type: Task type (if known)
            top_k: Number of experiences to retrieve
            include_meta: Whether to include meta-level experiences
            diversity_penalty: MMR diversity parameter (0=pure relevance, 1=pure diversity)

        Returns:
            List of top-k most relevant experiences
        """
        candidates = []

        # Always include meta-level experiences
        if include_meta:
            candidates.extend(self.meta_experiences)

        # Add domain-level if domain is known
        if domain:
            candidates.extend(self.domain_experiences.get(domain, []))

        # Add task-level if both domain and task_type are known
        if domain and task_type:
            candidates.extend(self.task_experiences.get((domain, task_type), []))

        # If no domain/task specified, use all experiences
        if not candidates:
            candidates = self.all_experiences.copy()

        if not candidates:
            return []

        # Compute problem embedding
        problem_embedding = self.embedding_fn(problem)

        # Compute similarity scores
        similarities = []
        for exp in candidates:
            sim = self._cosine_similarity(problem_embedding, exp.embedding)
            # Boost by effectiveness score
            adjusted_sim = sim * (1 + 0.5 * exp.effectiveness_score)
            similarities.append((exp, adjusted_sim))

        # Apply MMR (Maximum Marginal Relevance) for diversity
        selected = self._mmr_selection(
            similarities,
            top_k=top_k,
            lambda_param=1 - diversity_penalty
        )

        return selected

    def retrieve_by_difficulty(
        self,
        problem: str,
        difficulty: str,  # "easy", "medium", "hard"
        domain: Optional[str] = None
    ) -> List[Experience]:
        """
        Retrieve experiences with difficulty-adaptive top-k.

        Easier problems need fewer experiences; harder problems need more.
        """
        top_k_map = {
            "easy": 3,
            "medium": 5,
            "hard": 8
        }

        top_k = top_k_map.get(difficulty, 5)

        return self.retrieve_experiences(
            problem=problem,
            domain=domain,
            top_k=top_k
        )

    def update_effectiveness(
        self,
        experience: Experience,
        success: bool
    ):
        """Update effectiveness score based on usage outcome."""
        experience.usage_count += 1

        # Update success rate with exponential moving average
        alpha = 0.3  # Learning rate
        new_value = 1.0 if success else 0.0
        experience.success_rate = (
            alpha * new_value +
            (1 - alpha) * experience.success_rate
        )

        # Effectiveness combines success rate and usage frequency
        # More usage + high success rate → higher effectiveness
        usage_factor = min(experience.usage_count / 10.0, 1.0)  # Cap at 10 uses
        experience.effectiveness_score = (
            0.7 * experience.success_rate +
            0.3 * usage_factor
        )

    def get_statistics(self) -> Dict:
        """Get library statistics."""
        return {
            "total_experiences": len(self.all_experiences),
            "meta_experiences": len(self.meta_experiences),
            "domains": len(self.domain_experiences),
            "domain_breakdown": {
                domain: len(exps)
                for domain, exps in self.domain_experiences.items()
            },
            "avg_effectiveness": np.mean([
                exp.effectiveness_score for exp in self.all_experiences
            ]) if self.all_experiences else 0.0,
            "most_effective": sorted(
                self.all_experiences,
                key=lambda e: e.effectiveness_score,
                reverse=True
            )[:5]
        }

    def export_to_json(self, filepath: str):
        """Export library to JSON file."""
        data = {
            "meta": [self._exp_to_dict(e) for e in self.meta_experiences],
            "domain": {
                domain: [self._exp_to_dict(e) for e in exps]
                for domain, exps in self.domain_experiences.items()
            },
            "task": {
                f"{domain}:{task}": [self._exp_to_dict(e) for e in exps]
                for (domain, task), exps in self.task_experiences.items()
            }
        }

        with open(filepath, 'w') as f:
            json.dump(data, f, indent=2)

    def import_from_json(self, filepath: str):
        """Import library from JSON file."""
        with open(filepath, 'r') as f:
            data = json.load(f)

        # Import meta experiences
        for exp_dict in data.get("meta", []):
            self.add_experience(
                content=exp_dict["content"],
                level="meta",
                tags=set(exp_dict.get("tags", []))
            )

        # Import domain experiences
        for domain, exps in data.get("domain", {}).items():
            for exp_dict in exps:
                self.add_experience(
                    content=exp_dict["content"],
                    level="domain",
                    domain=domain,
                    tags=set(exp_dict.get("tags", []))
                )

        # Import task experiences
        for key, exps in data.get("task", {}).items():
            domain, task = key.split(":", 1)
            for exp_dict in exps:
                self.add_experience(
                    content=exp_dict["content"],
                    level="task",
                    domain=domain,
                    task_type=task,
                    tags=set(exp_dict.get("tags", []))
                )

    def _rebuild_index(self):
        """Rebuild flat embedding index for fast retrieval."""
        if not self.all_experiences:
            self.experience_embeddings = None
            return

        self.experience_embeddings = np.vstack([
            exp.embedding for exp in self.all_experiences
        ])

    def _mmr_selection(
        self,
        candidates: List[Tuple[Experience, float]],
        top_k: int,
        lambda_param: float = 0.7
    ) -> List[Experience]:
        """
        Maximum Marginal Relevance selection for diversity.

        MMR = λ * Relevance - (1-λ) * max Similarity to already selected
        """
        if not candidates:
            return []

        # Sort by initial relevance
        candidates = sorted(candidates, key=lambda x: x[1], reverse=True)

        selected = []
        remaining = candidates.copy()

        # Select first (most relevant)
        selected.append(remaining[0][0])
        remaining.pop(0)

        # Iteratively select based on MMR score
        while len(selected) < top_k and remaining:
            best_score = -float('inf')
            best_idx = 0

            for idx, (exp, relevance) in enumerate(remaining):
                # Compute max similarity to already selected
                max_sim = max([
                    self._cosine_similarity(exp.embedding, s.embedding)
                    for s in selected
                ]) if selected else 0

                # MMR score
                mmr_score = lambda_param * relevance - (1 - lambda_param) * max_sim

                if mmr_score > best_score:
                    best_score = mmr_score
                    best_idx = idx

            selected.append(remaining[best_idx][0])
            remaining.pop(best_idx)

        return selected

    def _cosine_similarity(self, vec1: np.ndarray, vec2: np.ndarray) -> float:
        """Compute cosine similarity between two vectors."""
        if vec1 is None or vec2 is None:
            return 0.0

        dot_product = np.dot(vec1, vec2)
        norm1 = np.linalg.norm(vec1)
        norm2 = np.linalg.norm(vec2)

        if norm1 == 0 or norm2 == 0:
            return 0.0

        return dot_product / (norm1 * norm2)

    def _simple_embedding(self, text: str) -> np.ndarray:
        """
        Simple embedding function using TF-IDF-like approach.

        In practice, replace with sentence-transformers or OpenAI embeddings.
        """
        # Tokenize
        words = text.lower().split()

        # Create vocabulary from all experiences (simplified)
        vocab = set()
        for exp in self.all_experiences:
            vocab.update(exp.content.lower().split())

        if not vocab:
            vocab = set(words)

        vocab_list = sorted(vocab)
        vocab_to_idx = {word: idx for idx, word in enumerate(vocab_list)}

        # Create vector
        vector = np.zeros(max(len(vocab_to_idx), 100))  # Min size 100

        for word in words:
            if word in vocab_to_idx:
                vector[vocab_to_idx[word]] += 1

        # Normalize
        norm = np.linalg.norm(vector)
        if norm > 0:
            vector = vector / norm

        return vector

    def _exp_to_dict(self, exp: Experience) -> Dict:
        """Convert Experience to dict for JSON serialization."""
        return {
            "content": exp.content,
            "level": exp.level,
            "domain": exp.domain,
            "task_type": exp.task_type,
            "effectiveness_score": exp.effectiveness_score,
            "success_rate": exp.success_rate,
            "usage_count": exp.usage_count,
            "tags": list(exp.tags)
        }


class ProblemClassifier:
    """Classifies problems to determine domain and task type for retrieval."""

    def __init__(self, llm_client=None):
        self.llm = llm_client

    def classify(self, problem: str) -> Tuple[str, str, str]:
        """
        Classify a problem.

        Returns:
            (domain, task_type, difficulty)
        """
        # Simple keyword-based classification (can be enhanced with LLM)
        problem_lower = problem.lower()

        # Detect domain
        domain = "general"
        if any(kw in problem_lower for kw in ["equation", "algebra", "geometry", "calculus", "math"]):
            domain = "math"
        elif any(kw in problem_lower for kw in ["code", "function", "algorithm", "programming"]):
            domain = "code"
        elif any(kw in problem_lower for kw in ["search", "web", "information", "query"]):
            domain = "web"

        # Detect task type (domain-specific)
        task_type = "general"
        if domain == "math":
            if "quadratic" in problem_lower or "polynomial" in problem_lower:
                task_type = "algebra"
            elif "triangle" in problem_lower or "circle" in problem_lower:
                task_type = "geometry"
            elif "derivative" in problem_lower or "integral" in problem_lower:
                task_type = "calculus"

        # Detect difficulty (simplified)
        difficulty = "medium"
        if len(problem.split()) < 20:
            difficulty = "easy"
        elif len(problem.split()) > 50:
            difficulty = "hard"

        return domain, task_type, difficulty


# Example usage
if __name__ == "__main__":
    print("=== Hierarchical Retrieval-Augmented Prior Demo ===\n")

    # Create library
    library = HierarchicalExperienceLibrary()

    # Add meta-level experiences (universal strategies)
    print("Adding meta-level experiences...")
    library.add_experience(
        "Break complex problems into smaller, manageable steps",
        level="meta",
        tags={"decomposition", "problem-solving"}
    )
    library.add_experience(
        "Verify each intermediate result before proceeding to the next step",
        level="meta",
        tags={"verification", "accuracy"}
    )
    library.add_experience(
        "If stuck, try working backwards from the desired outcome",
        level="meta",
        tags={"strategy", "backwards-reasoning"}
    )

    # Add domain-level experiences (math domain)
    print("Adding domain-level experiences (math)...")
    library.add_experience(
        "Use algebraic manipulation to simplify equations before solving",
        level="domain",
        domain="math",
        tags={"algebra", "simplification"}
    )
    library.add_experience(
        "Draw a diagram to visualize geometric relationships",
        level="domain",
        domain="math",
        tags={"geometry", "visualization"}
    )

    # Add task-level experiences (algebra tasks)
    print("Adding task-level experiences (math/algebra)...")
    library.add_experience(
        "When solving quadratic equations, first check if factorization is possible",
        level="task",
        domain="math",
        task_type="algebra",
        tags={"quadratic", "factorization"}
    )
    library.add_experience(
        "For systems of linear equations, consider using substitution or elimination method",
        level="task",
        domain="math",
        task_type="algebra",
        tags={"linear-system", "methods"}
    )

    # Add domain-level experiences (code domain)
    print("Adding domain-level experiences (code)...")
    library.add_experience(
        "Write test cases before implementing the solution",
        level="domain",
        domain="code",
        tags={"testing", "tdd"}
    )

    # Show statistics
    print("\n" + "="*60)
    stats = library.get_statistics()
    print(f"Library Statistics:")
    print(f"  Total experiences: {stats['total_experiences']}")
    print(f"  Meta-level: {stats['meta_experiences']}")
    print(f"  Domains: {stats['domains']}")
    print(f"  Domain breakdown: {stats['domain_breakdown']}")

    # Test retrieval
    print("\n" + "="*60)
    print("Test Case 1: Specific math/algebra problem")
    problem1 = "Solve the quadratic equation x^2 + 5x + 6 = 0"

    # Classify problem
    classifier = ProblemClassifier()
    domain, task_type, difficulty = classifier.classify(problem1)
    print(f"Problem: {problem1}")
    print(f"Classified as: domain={domain}, task_type={task_type}, difficulty={difficulty}")

    # Retrieve relevant experiences
    retrieved = library.retrieve_by_difficulty(
        problem=problem1,
        difficulty=difficulty,
        domain=domain
    )

    print(f"\nRetrieved {len(retrieved)} experiences:")
    for i, exp in enumerate(retrieved, 1):
        print(f"{i}. [{exp.level:6s}] {exp.content}")

    # Test Case 2: General problem
    print("\n" + "="*60)
    print("Test Case 2: General problem without specific domain")
    problem2 = "How can I optimize my workflow?"

    domain2, task_type2, difficulty2 = classifier.classify(problem2)
    print(f"Problem: {problem2}")
    print(f"Classified as: domain={domain2}, task_type={task_type2}, difficulty={difficulty2}")

    retrieved2 = library.retrieve_experiences(
        problem=problem2,
        domain=domain2,
        top_k=3
    )

    print(f"\nRetrieved {len(retrieved2)} experiences:")
    for i, exp in enumerate(retrieved2, 1):
        print(f"{i}. [{exp.level:6s}] {exp.content}")

    # Simulate effectiveness updates
    print("\n" + "="*60)
    print("Simulating effectiveness tracking...")
    for exp in retrieved[:2]:
        library.update_effectiveness(exp, success=True)
        print(f"Updated: {exp.content[:50]}... → effectiveness={exp.effectiveness_score:.3f}")

    # Export library
    print("\n" + "="*60)
    print("Exporting library to JSON...")
    library.export_to_json("/tmp/experience_library.json")
    print("✓ Exported to /tmp/experience_library.json")

    print("\n✓ Hierarchical Retrieval-Augmented Prior demo complete!")
    print("\nKey Benefits:")
    print("  - Organizes experiences hierarchically (meta → domain → task)")
    print("  - Retrieves only relevant experiences per problem")
    print("  - Tracks effectiveness to improve over time")
    print("  - Supports cross-domain transfer via meta-experiences")
    print("  - Scalable to large experience libraries")
