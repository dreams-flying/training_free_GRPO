"""
Meta-Learning and Cross-Domain Transfer Module

Enables:
1. Extraction of domain-agnostic meta-experiences
2. Transfer learning across domains (math → web, etc.)
3. Few-shot adaptation to new domains
4. Hierarchical experience organization
"""

import json
import os
from typing import List, Dict, Tuple
from collections import defaultdict
from training_free_grpo.llm import LLM


class MetaExperienceExtractor:
    """
    Extracts high-level, domain-agnostic meta-experiences.

    Meta-experiences are principles that apply across domains,
    e.g., "Break complex problems into steps" applies to both math and web search.
    """

    def __init__(self):
        self.llm = LLM()

    def extract_meta_experiences(
        self,
        domain_experiences: Dict[str, Dict[str, str]]  # {domain: {id: exp}}
    ) -> Dict[str, str]:
        """
        Extract meta-experiences from multiple domain-specific experience sets.

        Args:
            domain_experiences: Dict mapping domain name to experience dict

        Returns:
            Dict of meta-experiences {meta_id: meta_exp}
        """
        print(f"[Meta-Learning] Extracting meta-experiences from {len(domain_experiences)} domains...")

        # Collect all experiences grouped by domain
        domain_texts = {}
        for domain, experiences in domain_experiences.items():
            domain_texts[domain] = list(experiences.values())

        # Build prompt for meta-extraction
        prompt = self._build_meta_extraction_prompt(domain_texts)

        try:
            response = self.llm.chat(prompt, temperature=0.2, max_tokens=2048)
            meta_experiences = self._parse_meta_experiences(response)

            print(f"[Meta-Learning] Extracted {len(meta_experiences)} meta-experiences")
            return meta_experiences
        except Exception as e:
            print(f"Warning: Meta-experience extraction failed: {e}")
            return {}

    def _build_meta_extraction_prompt(self, domain_texts: Dict[str, List[str]]) -> str:
        """Build prompt for extracting meta-experiences."""

        experiences_text = ""
        for domain, experiences in domain_texts.items():
            experiences_text += f"\n=== {domain.upper()} DOMAIN ===\n"
            for i, exp in enumerate(experiences[:10], 1):  # Limit to 10 per domain
                experiences_text += f"{i}. {exp}\n"

        prompt = f"""
You are a meta-learning system analyzing problem-solving experiences across multiple domains.

{experiences_text}

TASK: Extract HIGH-LEVEL, DOMAIN-AGNOSTIC meta-experiences that apply across all domains.

A good meta-experience should:
1. Apply to MULTIPLE domains (not specific to math, web, etc.)
2. Capture GENERAL problem-solving strategies
3. Be ACTIONABLE and provide clear guidance
4. Be at a HIGHER level of abstraction than domain-specific experiences

Examples of meta-experiences:
- "Break complex problems into smaller, manageable subproblems"
- "Verify assumptions before proceeding with a solution"
- "When stuck, try restating the problem in different terms"
- "Always validate your final answer against the original constraints"

Extract 5-7 meta-experiences that capture the COMMON WISDOM across these domains.

Format your response as:
META-1: [meta-experience text]
META-2: [meta-experience text]
...
"""

        return prompt

    def _parse_meta_experiences(self, response: str) -> Dict[str, str]:
        """Parse meta-experiences from LLM response."""
        meta_exp = {}
        lines = response.strip().split('\n')

        for line in lines:
            line = line.strip()
            if line.startswith('META-') or line.startswith('META '):
                if ':' in line:
                    # Extract ID and text
                    parts = line.split(':', 1)
                    id_part = parts[0].strip()
                    text = parts[1].strip()

                    # Clean ID
                    meta_id = id_part.replace('META-', 'M').replace('META ', 'M')
                    if len(text) > 20:
                        meta_exp[meta_id] = text

        return meta_exp

    def specialize_meta_experience(
        self,
        meta_experience: str,
        target_domain: str,
        example_problems: List[str] = None
    ) -> str:
        """
        Specialize a meta-experience for a specific domain.

        Args:
            meta_experience: General meta-experience text
            target_domain: Target domain (e.g., "math", "web", "code")
            example_problems: Optional example problems from target domain

        Returns:
            Domain-specific version of the meta-experience
        """
        examples_text = ""
        if example_problems:
            examples_text = "\n\nEXAMPLE PROBLEMS IN THIS DOMAIN:\n"
            for i, prob in enumerate(example_problems[:3], 1):
                examples_text += f"{i}. {prob[:150]}...\n"

        prompt = f"""
Adapt the following GENERAL meta-experience to the {target_domain} domain.

META-EXPERIENCE: {meta_experience}

TARGET DOMAIN: {target_domain}
{examples_text}

TASK: Create a domain-specific version that:
1. Keeps the core principle of the meta-experience
2. Uses terminology and concepts relevant to {target_domain}
3. Provides concrete guidance for {target_domain} problems
4. Remains concise (1-2 sentences)

Respond with ONLY the adapted experience text.
"""

        try:
            specialized = self.llm.chat(prompt, temperature=0.0, max_tokens=256)
            return specialized.strip()
        except Exception as e:
            print(f"Warning: Specialization failed: {e}")
            return meta_experience  # Return original if specialization fails


class CrossDomainTransfer:
    """
    Transfers experiences from source domain to target domain.

    Useful for:
    - Bootstrapping a new domain with limited data
    - Leveraging successful strategies from one domain in another
    - Finding analogies between different problem types
    """

    def __init__(self):
        self.llm = LLM()

    def transfer_experiences(
        self,
        source_experiences: Dict[str, str],
        source_domain: str,
        target_domain: str,
        target_problems: List[str] = None
    ) -> Dict[str, str]:
        """
        Transfer experiences from source domain to target domain.

        Args:
            source_experiences: Experiences from source domain
            source_domain: Name of source domain
            target_domain: Name of target domain
            target_problems: Sample problems from target domain

        Returns:
            Transferred experiences adapted to target domain
        """
        print(f"[Transfer] Transferring {len(source_experiences)} experiences "
              f"from {source_domain} → {target_domain}")

        transferred = {}

        for exp_id, exp_text in source_experiences.items():
            # Assess transferability
            is_transferable = self._assess_transferability(
                exp_text, source_domain, target_domain
            )

            if is_transferable:
                # Transfer and adapt
                adapted = self._adapt_experience(
                    exp_text, source_domain, target_domain, target_problems
                )
                if adapted:
                    new_id = f"T{exp_id}"  # 'T' for Transferred
                    transferred[new_id] = adapted
                    print(f"  ✓ Transferred {exp_id} → {new_id}")
            else:
                print(f"  ✗ Skipped {exp_id} (not transferable)")

        print(f"[Transfer] Successfully transferred {len(transferred)} experiences")
        return transferred

    def _assess_transferability(
        self,
        experience: str,
        source_domain: str,
        target_domain: str
    ) -> bool:
        """
        Assess whether an experience can be transferred to target domain.

        Returns:
            True if transferable, False otherwise
        """
        # Simple heuristic: check for domain-specific keywords
        domain_specific_keywords = {
            "math": ["equation", "derivative", "integral", "proof", "theorem", "formula"],
            "web": ["search", "url", "website", "browser", "query", "link"],
            "code": ["function", "variable", "loop", "debug", "compile", "syntax"],
        }

        exp_lower = experience.lower()

        # Count domain-specific terms from source domain
        source_keywords = domain_specific_keywords.get(source_domain, [])
        specific_count = sum(1 for kw in source_keywords if kw in exp_lower)

        # If experience has many domain-specific terms, it's harder to transfer
        # If it's general (few specific terms), it's more transferable
        return specific_count <= 1

    def _adapt_experience(
        self,
        experience: str,
        source_domain: str,
        target_domain: str,
        target_problems: List[str] = None
    ) -> str:
        """Adapt an experience from source to target domain."""

        problems_text = ""
        if target_problems:
            problems_text = "\n\nSAMPLE TARGET PROBLEMS:\n"
            for i, prob in enumerate(target_problems[:3], 1):
                problems_text += f"{i}. {prob[:120]}...\n"

        prompt = f"""
Adapt the following experience from {source_domain} domain to {target_domain} domain.

SOURCE EXPERIENCE ({source_domain}): {experience}

TARGET DOMAIN: {target_domain}
{problems_text}

TASK: Rewrite this experience to be relevant for {target_domain} problems.

Guidelines:
1. Keep the CORE STRATEGY or principle
2. Replace domain-specific terms with {target_domain}-appropriate ones
3. Make it actionable for {target_domain} problems
4. Keep it concise (1-2 sentences)

If the experience cannot be meaningfully adapted, respond with "NOT_TRANSFERABLE".

Respond with ONLY the adapted experience text.
"""

        try:
            adapted = self.llm.chat(prompt, temperature=0.0, max_tokens=256)
            adapted = adapted.strip()

            if "NOT_TRANSFERABLE" in adapted:
                return None
            return adapted
        except Exception as e:
            print(f"Warning: Adaptation failed: {e}")
            return None


class HierarchicalExperienceOrganizer:
    """
    Organizes experiences in a hierarchy:
    - Meta-experiences (highest level, domain-agnostic)
    - Domain experiences (medium level, domain-specific)
    - Task-specific experiences (lowest level, problem-type specific)
    """

    def __init__(self):
        self.meta_experiences = {}
        self.domain_experiences = defaultdict(dict)  # {domain: {id: exp}}
        self.task_experiences = defaultdict(lambda: defaultdict(dict))  # {domain: {task: {id: exp}}}

    def add_meta_experience(self, meta_id: str, meta_exp: str):
        """Add a meta-level experience."""
        self.meta_experiences[meta_id] = meta_exp

    def add_domain_experience(self, domain: str, exp_id: str, exp_text: str):
        """Add a domain-specific experience."""
        self.domain_experiences[domain][exp_id] = exp_text

    def add_task_experience(self, domain: str, task: str, exp_id: str, exp_text: str):
        """Add a task-specific experience."""
        self.task_experiences[domain][task][exp_id] = exp_text

    def get_relevant_experiences(
        self,
        domain: str,
        task: str = None,
        include_meta: bool = True
    ) -> Dict[str, str]:
        """
        Get all relevant experiences for a domain/task.

        Combines experiences from multiple levels of the hierarchy.

        Args:
            domain: Target domain
            task: Optional specific task type
            include_meta: Whether to include meta-experiences

        Returns:
            Combined experience dict
        """
        experiences = {}

        # Level 1: Meta-experiences (if requested)
        if include_meta:
            experiences.update(self.meta_experiences)

        # Level 2: Domain experiences
        if domain in self.domain_experiences:
            experiences.update(self.domain_experiences[domain])

        # Level 3: Task-specific experiences
        if task and domain in self.task_experiences and task in self.task_experiences[domain]:
            experiences.update(self.task_experiences[domain][task])

        return experiences

    def save_hierarchy(self, filepath: str):
        """Save hierarchical structure to file."""
        data = {
            "meta_experiences": self.meta_experiences,
            "domain_experiences": dict(self.domain_experiences),
            "task_experiences": {
                domain: dict(tasks)
                for domain, tasks in self.task_experiences.items()
            }
        }
        with open(filepath, 'w') as f:
            json.dump(data, f, indent=2)
        print(f"[Hierarchy] Saved to {filepath}")

    def load_hierarchy(self, filepath: str):
        """Load hierarchical structure from file."""
        if os.path.exists(filepath):
            with open(filepath, 'r') as f:
                data = json.load(f)
                self.meta_experiences = data.get("meta_experiences", {})
                self.domain_experiences = defaultdict(dict, data.get("domain_experiences", {}))
                self.task_experiences = defaultdict(
                    lambda: defaultdict(dict),
                    data.get("task_experiences", {})
                )
            print(f"[Hierarchy] Loaded {len(self.meta_experiences)} meta, "
                  f"{len(self.domain_experiences)} domain experiences")


if __name__ == "__main__":
    print("=== Meta-Learning Demo ===\n")

    # Create sample experiences from different domains
    math_experiences = {
        "M1": "When solving optimization problems, find critical points by taking derivatives",
        "M2": "Always verify solutions by substituting back into the original equation",
        "M3": "Break complex proofs into smaller lemmas",
    }

    web_experiences = {
        "W1": "Start with broad searches, then refine with specific terms",
        "W2": "Verify information by cross-referencing multiple authoritative sources",
        "W3": "Break complex queries into simpler sub-queries",
    }

    # Extract meta-experiences
    extractor = MetaExperienceExtractor()
    domain_exps = {"math": math_experiences, "web": web_experiences}
    meta_exps = extractor.extract_meta_experiences(domain_exps)

    print("Extracted Meta-Experiences:")
    for id, exp in meta_exps.items():
        print(f"  {id}: {exp}")

    # Transfer from math to web
    print("\n=== Cross-Domain Transfer Demo ===\n")

    transfer = CrossDomainTransfer()
    target_problems = [
        "Find information about the 2024 Nobel Prize winners",
        "What is the current GDP of Japan?",
    ]

    transferred = transfer.transfer_experiences(
        math_experiences,
        source_domain="math",
        target_domain="web",
        target_problems=target_problems
    )

    print("\nTransferred Experiences (math → web):")
    for id, exp in transferred.items():
        print(f"  {id}: {exp}")

    # Hierarchical organization
    print("\n=== Hierarchical Organization Demo ===\n")

    organizer = HierarchicalExperienceOrganizer()

    # Add experiences at different levels
    organizer.add_meta_experience("M1", "Break problems into smaller parts")
    organizer.add_meta_experience("M2", "Verify your solutions")

    organizer.add_domain_experience("math", "D1", "Use calculus for optimization")
    organizer.add_domain_experience("web", "D2", "Use boolean operators for searches")

    organizer.add_task_experience("math", "derivatives", "T1", "Apply chain rule for compositions")
    organizer.add_task_experience("web", "fact-finding", "T2", "Check official sources first")

    # Retrieve relevant experiences for a specific task
    print("Relevant experiences for math/derivatives:")
    relevant = organizer.get_relevant_experiences("math", task="derivatives", include_meta=True)
    for id, exp in relevant.items():
        print(f"  {id}: {exp}")

    # Save and load
    organizer.save_hierarchy("demo_hierarchy.json")
    print("\nHierarchy saved successfully")
