"""
Enhanced Training Script with Improved Experience Learning

Integrates:
1. Experience Quality Assessment
2. Dynamic Experience Retrieval
3. Incremental Learning
4. Empirical Effectiveness Tracking

Usage:
    python training_free_grpo/train_enhanced.py \
        --mode agent \
        --domain math \
        --experiment_name enhanced_exp \
        --dataset DAPO-Math-17k \
        --dataset_truncate 100 \
        --use_quality_filter \
        --use_retrieval \
        --use_incremental \
        --quality_threshold 0.6 \
        --retrieval_top_k 5
"""

# Windows compatibility: must be first import
try:
    from training_free_grpo import _windows_compat
except ImportError:
    pass

import argparse
import asyncio
import copy
import json
import os
import random

from training_free_grpo.main import rollout_dataset, load_rollouts
from training_free_grpo.experience_quality import (
    ExperienceQualityAssessor,
    EmpiricalEffectivenessTracker
)
from training_free_grpo.experience_retrieval import DifficultyAdaptiveRetriever
from training_free_grpo.incremental_learning import IncrementalExperienceUpdater
from utu.agents import SimpleAgent
from utu.config import ConfigLoader

random.seed(42)


async def main(args):
    # Set up domain-specific variables
    if args.domain == "math":
        from training_free_grpo.math.dataset import load_data
        from training_free_grpo.math.verify import verify_func
        from training_free_grpo.math.prompts import PROBLEM_WITH_EXPERIENCE_TEMPLATE
        from training_free_grpo.math.experience import ExperienceUpdater
        config_name = "simple/math_agent.yaml"
    elif args.domain == "web":
        from training_free_grpo.web.dataset import load_data
        from training_free_grpo.web.verify import verify_func
        from training_free_grpo.web.prompts import PROBLEM_WITH_EXPERIENCE_TEMPLATE
        from training_free_grpo.web.experience import ExperienceUpdater
        config_name = "simple/base_search.yaml"
    else:
        raise ValueError(f"Unsupported domain: {args.domain}")

    # Create experiment directory
    experiment_dir = os.path.join("data", args.domain, "train", args.experiment_name)
    os.makedirs(experiment_dir, exist_ok=True)

    # Initialize enhancement modules
    print("=" * 70)
    print("ENHANCED TRAINING-FREE GRPO")
    print("=" * 70)

    quality_assessor = None
    effectiveness_tracker = None
    experience_retriever = None
    incremental_updater = None

    if args.use_quality_filter:
        print(f"✓ Quality Filter: threshold={args.quality_threshold}")
        quality_assessor = ExperienceQualityAssessor(
            quality_threshold=args.quality_threshold
        )

    if args.use_effectiveness_tracking:
        print(f"✓ Effectiveness Tracking: enabled")
        effectiveness_path = os.path.join(experiment_dir, "effectiveness.json")
        effectiveness_tracker = EmpiricalEffectivenessTracker(save_path=effectiveness_path)

    if args.use_retrieval:
        print(f"✓ Dynamic Retrieval: top_k={args.retrieval_top_k}")
        experience_retriever = DifficultyAdaptiveRetriever(
            top_k=args.retrieval_top_k,
            use_semantic_sim=args.use_semantic_retrieval,
            diversity_penalty=0.2
        )

    if args.use_incremental:
        print(f"✓ Incremental Learning: update_freq={args.incremental_update_freq}")
        incremental_path = os.path.join(experiment_dir, "incremental_experiences.json")
        incremental_updater = IncrementalExperienceUpdater(
            window_size=100,
            update_frequency=args.incremental_update_freq,
            save_path=incremental_path
        )

    print("=" * 70)
    print()

    # Set up the agent
    if args.mode == "prompt":
        worker_agent = None
    elif args.mode == "agent":
        config = ConfigLoader.load_agent_config(config_name)
        config.model.model_settings.temperature = args.rollout_temperature
        worker_agent = SimpleAgent(config=config)
        await worker_agent.build()
    else:
        raise ValueError(f"Unsupported inference mode: {args.mode}")

    # Load the dataset
    train_data = load_data(args.dataset)
    print(f"Loaded {len(train_data)} records from dataset")
    if args.dataset_truncate is not None:
        print(f"- truncated to {args.dataset_truncate}")
        train_data = train_data[:args.dataset_truncate]
    assert len(train_data) % args.batchsize == 0

    # Set up the stats
    stats_filename = os.path.join(experiment_dir, "stats.json")
    if os.path.exists(stats_filename):
        stats = json.load(open(stats_filename))
    else:
        stats = {}

    # Train
    for epoch in range(args.epochs):
        # Init
        print("=" * 70)
        print(f"Epoch {epoch}")
        print("=" * 70)
        cur_epoch_dir = os.path.join(experiment_dir, f"epoch_{epoch}")
        os.makedirs(cur_epoch_dir, exist_ok=True)

        # Check if shuffled data already exists for this epoch
        shuffled_filename = os.path.join(cur_epoch_dir, "shuffled_data.jsonl")
        if os.path.exists(shuffled_filename):
            shuffled_data = []
            with open(shuffled_filename) as f:
                for line in f:
                    shuffled_data.append(json.loads(line))
            print(f"Loaded {len(shuffled_data)} records from shuffled data")
        else:
            print(f"Shuffling data ...")
            shuffled_data = copy.deepcopy(train_data)
            random.shuffle(shuffled_data)
            with open(shuffled_filename, "w") as f:
                for each in shuffled_data:
                    f.write(json.dumps(each) + "\n")

        # for each batch
        num_batches = len(shuffled_data) // args.batchsize
        for batch_idx in range(num_batches):
            step = epoch * num_batches + batch_idx
            if f"step_{step}" not in stats:
                stats[f"step_{step}"] = {"epoch": epoch, "batch": batch_idx, "complete": False}
            elif stats[f"step_{step}"]["complete"]:
                continue

            # Init
            print(f"\nStep {step} (Epoch {epoch}, Batch {batch_idx})")
            cur_step_dir = os.path.join(experiment_dir, f"step_{step}")
            os.makedirs(cur_step_dir, exist_ok=True)

            # Get current batch data
            batch_data = copy.deepcopy(shuffled_data[batch_idx * args.batchsize: (batch_idx + 1) * args.batchsize])

            # Load existing rollouts
            rollout_filename = os.path.join(cur_step_dir, "rollout.jsonl")
            rollouts = load_rollouts(rollout_filename)

            # Retrieve experiences for this batch
            if step > 0:
                experience_filename = os.path.join(
                    "data", args.domain, "train", args.experiment_name,
                    f"step_{step}/experiences.json"
                )
                if os.path.exists(experience_filename):
                    experiences = json.load(open(experience_filename))
                else:
                    experiences = {}
            else:
                experiences = {}

            # ENHANCEMENT 1: Apply quality filtering if enabled
            if quality_assessor and experiences:
                print(f"\n[Quality Filter] Filtering {len(experiences)} experiences...")
                filtered_experiences = quality_assessor.filter_experiences(
                    experiences,
                    domain=args.domain,
                    min_quality=args.quality_threshold
                )
                experiences = {id: exp for id, (exp, score) in filtered_experiences.items()}
                print(f"[Quality Filter] Kept {len(experiences)} high-quality experiences")

            # Prepare effectiveness scores for retrieval
            effectiveness_scores = None
            if effectiveness_tracker and experiences:
                effectiveness_scores = {}
                for exp_id in experiences.keys():
                    metrics = effectiveness_tracker.get_effectiveness(exp_id)
                    # Use success rate as score, default to 0.5 for new experiences
                    effectiveness_scores[exp_id] = metrics.get("success_rate", 0.5)

            # Format the batch data with experiences
            formatted_batch_data = []
            for each in batch_data:
                problem = each["problem"]

                # ENHANCEMENT 2: Use dynamic retrieval if enabled
                if experience_retriever and experiences:
                    retrieved = experience_retriever.retrieve_experiences(
                        problem,
                        experiences,
                        effectiveness_scores=effectiveness_scores
                    )
                    formatted_experiences = experience_retriever.format_retrieved_experiences(retrieved)
                    # Track which experiences were used
                    used_exp_ids = [exp_id for exp_id, _, _ in retrieved]
                else:
                    # Use all experiences (original behavior)
                    formatted_experiences = "\n".join([f"[{i}]. {e}" for i, e in experiences.items()])
                    used_exp_ids = list(experiences.keys())

                prompt = PROBLEM_WITH_EXPERIENCE_TEMPLATE.format(
                    experiences=formatted_experiences if formatted_experiences else "None",
                    problem=problem,
                ) if experiences else problem

                formatted_batch_data.append({
                    **each,
                    "prompt": prompt,
                    "used_experience_ids": used_exp_ids
                })

            # Duplicate for GRPO
            print(f"GRPO rollout number={args.grpo_n}")
            formatted_batch_data = formatted_batch_data * args.grpo_n

            # Rollout the dataset
            rollouts, rollout_stats = await rollout_dataset(
                worker_agent=worker_agent,
                data=formatted_batch_data,
                rollouts=rollouts,
                verify_func=verify_func,
                rollout_filename=rollout_filename,
                rollout_concurrency=args.rollout_concurrency,
                task_timeout=args.task_timeout,
                temperature=args.rollout_temperature,
                max_tokens=args.rollout_max_tokens,
            )
            stats[f"step_{step}"]["rollout"] = rollout_stats

            # ENHANCEMENT 3: Track effectiveness if enabled
            if effectiveness_tracker:
                print(f"\n[Effectiveness] Recording {len(rollouts)} rollout results...")
                for rollout in rollouts:
                    if "used_experience_ids" in rollout:
                        effectiveness_tracker.record_usage(
                            rollout["used_experience_ids"],
                            rollout.get("reward", 0.0)
                        )
                effectiveness_tracker.save()

                # Optionally prune ineffective experiences
                if args.prune_ineffective and step > 0 and step % 3 == 0:
                    print(f"[Effectiveness] Pruning ineffective experiences...")
                    experiences = effectiveness_tracker.prune_ineffective_experiences(
                        experiences,
                        min_success_rate=0.3,
                        min_usage=10
                    )

            # ENHANCEMENT 4: Add to incremental learner if enabled
            if incremental_updater:
                print(f"\n[Incremental] Adding {len(rollouts)} rollouts to learner...")
                for rollout in rollouts:
                    if "problem" in rollout and "response" in rollout:
                        incremental_updater.add_rollout(
                            problem=rollout["problem"],
                            response=rollout["response"],
                            reward=rollout.get("reward", 0.0),
                            trajectory=rollout.get("trajectories", [])
                        )

            # Standard experience update
            next_step_dir = os.path.join(experiment_dir, f"step_{step + 1}")
            os.makedirs(next_step_dir, exist_ok=True)
            next_experience_filename = os.path.join(next_step_dir, "experiences.json")

            if os.path.exists(next_experience_filename):
                print(f"Experiences already exist for step {step}, skipping experience update")
            else:
                # Run standard experience updater
                print(f"\n[Standard Update] Generating experiences from rollouts...")
                new_experiences = ExperienceUpdater().run(
                    rollouts=rollouts,
                    experiences=experiences,
                    save_dir=cur_step_dir,
                    max_workers=args.rollout_concurrency,
                    given_ground_truth=True if args.given_ground_truth == "True" else False,
                    only_partial_correct=True if args.grpo_n > 1 else False,
                )

                # ENHANCEMENT: Merge with incremental experiences if enabled
                if incremental_updater and args.use_incremental:
                    incremental_exp = incremental_updater.get_experiences()
                    if incremental_exp:
                        print(f"[Incremental] Merging {len(incremental_exp)} incremental experiences...")
                        # Merge: incremental experiences take precedence for conflicts
                        new_experiences.update(incremental_exp)

                json.dump(new_experiences, open(next_experience_filename, "w"), indent=2)
                print(f"Saved {len(new_experiences)} experiences to {next_experience_filename}")

            # Save stats
            stats[f"step_{step}"]["complete"] = True
            json.dump(stats, open(stats_filename, "w"), indent=2)

    print("\n" + "=" * 70)
    print("TRAINING COMPLETE")
    print("=" * 70)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Enhanced Training-free GRPO")

    # Original arguments
    parser.add_argument("--mode", type=str, default="agent", required=True, choices=["prompt", "agent"])
    parser.add_argument("--domain", type=str, required=True, choices=["math", "web"])
    parser.add_argument("--experiment_name", type=str, required=True)
    parser.add_argument("--dataset", type=str, required=True)
    parser.add_argument("--dataset_truncate", type=int, default=None)
    parser.add_argument("--given_ground_truth", type=str, default="True")
    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--batchsize", type=int, default=64)
    parser.add_argument("--grpo_n", type=int, default=5)
    parser.add_argument("--rollout_concurrency", type=int, default=5)
    parser.add_argument("--rollout_temperature", type=float, default=0.7)
    parser.add_argument("--rollout_max_tokens", type=int, default=16384)
    parser.add_argument("--task_timeout", type=float, default=3600)

    # ENHANCEMENT: Quality filtering
    parser.add_argument("--use_quality_filter", action="store_true",
                        help="Enable experience quality filtering")
    parser.add_argument("--quality_threshold", type=float, default=0.6,
                        help="Minimum quality score to keep experience")

    # ENHANCEMENT: Dynamic retrieval
    parser.add_argument("--use_retrieval", action="store_true",
                        help="Enable dynamic experience retrieval")
    parser.add_argument("--retrieval_top_k", type=int, default=5,
                        help="Number of experiences to retrieve per problem")
    parser.add_argument("--use_semantic_retrieval", action="store_true",
                        help="Use semantic similarity (slower but more accurate)")

    # ENHANCEMENT: Effectiveness tracking
    parser.add_argument("--use_effectiveness_tracking", action="store_true",
                        help="Track empirical effectiveness of experiences")
    parser.add_argument("--prune_ineffective", action="store_true",
                        help="Prune experiences with low success rate")

    # ENHANCEMENT: Incremental learning
    parser.add_argument("--use_incremental", action="store_true",
                        help="Enable incremental experience learning")
    parser.add_argument("--incremental_update_freq", type=int, default=10,
                        help="Update incremental experiences every N rollouts")

    args = parser.parse_args()
    asyncio.run(main(args))
