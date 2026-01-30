#!/usr/bin/env python3
"""
Generate training and evaluation data using local vLLM server.
This script creates diverse question-answer pairs with step-by-step reasoning.
"""

import json
import random
import argparse
from pathlib import Path
from openai import OpenAI
from tqdm import tqdm
from typing import List, Dict

client = OpenAI(base_url="http://localhost:8000/v1", api_key="dummy")
MODEL_NAME: str = "glm-4.7"

# Topic categories for diverse data generation
TOPICS = [
    "Python programming",
    "Data structures and algorithms",
    "Machine learning concepts",
    "Web development",
    "Database design",
    "System design",
    "Mathematics and statistics",
    "Computer networks",
    "Operating systems",
    "Software engineering best practices",
    "API design",
    "Cloud computing",
    "Cybersecurity",
    "DevOps and CI/CD",
    "Testing and quality assurance",
]

TASK_TYPES = [
    "Write a function to",
    "Explain how to",
    "What is the difference between",
    "How would you optimize",
    "Debug this code:",
    "Design a system for",
    "Compare and contrast",
    "Implement an algorithm for",
    "What are the best practices for",
    "Solve this problem:",
]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Generate JSON training/eval data via an OpenAI-compatible vLLM server.")
    p.add_argument("--base_url", type=str, default="http://localhost:8000/v1", help="vLLM OpenAI base URL.")
    p.add_argument("--api_key", type=str, default="dummy", help="API key (vLLM usually ignores this).")
    p.add_argument(
        "--model",
        type=str,
        default="glm-4.7",
        help="Served model name (see `vllm serve --served-model-name`).",
    )
    p.add_argument("--train_samples", type=int, default=100, help="Number of training samples to generate.")
    p.add_argument("--eval_samples", type=int, default=20, help="Number of eval samples to generate.")
    p.add_argument(
        "--out_dir",
        type=str,
        default="data/generated/tooluse_data",
        help="Output directory (kept separate to avoid overwriting bundled datasets).",
    )
    p.add_argument("--seed", type=int, default=42, help="Random seed.")
    return p.parse_args()


def make_client(base_url: str, api_key: str) -> OpenAI:
    return OpenAI(base_url=base_url, api_key=api_key)


def generate_prompt(topic: str, task_type: str) -> str:
    """Generate a diverse prompt based on topic and task type."""
    prompt_template = f"""Generate a single technical question or coding task about {topic}.
The task should start with or relate to: "{task_type}"
Make it specific, practical, and suitable for a technical interview or learning scenario.
Output ONLY the question/task, nothing else."""

    response = client.chat.completions.create(
        model=MODEL_NAME,
        messages=[{"role": "user", "content": prompt_template}],
        temperature=0.9,
        max_tokens=150,
    )
    
    return response.choices[0].message.content.strip()


def generate_golden_response(prompt: str) -> List[str]:
    """Generate a step-by-step golden response for the given prompt."""
    response_template = f"""You are an expert technical instructor. Answer the following question with clear, step-by-step reasoning.

Question: {prompt}

Provide your answer as a series of clear steps or reasoning points. Show your thinking process, not just the final answer. If it involves code, include code blocks with explanations.

Break down your response into numbered steps or logical parts."""

    response = client.chat.completions.create(
        model=MODEL_NAME,
        messages=[{"role": "user", "content": response_template}],
        temperature=0.7,
        max_tokens=800,
    )
    
    content = response.choices[0].message.content.strip()
    
    # Split response into logical steps
    # Try to preserve existing structure if numbered
    lines = content.split('\n')
    steps = []
    current_step = []
    
    for line in lines:
        line = line.strip()
        if not line:
            if current_step:
                steps.append('\n'.join(current_step))
                current_step = []
            continue
        
        # Check if this is a new numbered step
        if any(line.startswith(f"{i}.") or line.startswith(f"{i})") for i in range(1, 20)):
            if current_step:
                steps.append('\n'.join(current_step))
            current_step = [line]
        else:
            current_step.append(line)
    
    # Add the last step
    if current_step:
        steps.append('\n'.join(current_step))
    
    # If no clear structure, split by paragraphs
    if len(steps) <= 1:
        steps = [s.strip() for s in content.split('\n\n') if s.strip()]
    
    # Ensure we have at least 2 steps
    if len(steps) < 2:
        steps = [content]
    
    return steps


def generate_dataset(num_samples: int, desc: str = "Generating") -> List[Dict]:
    """Generate a dataset with specified number of samples."""
    dataset = []
    
    for i in tqdm(range(num_samples), desc=desc):
        topic = random.choice(TOPICS)
        task_type = random.choice(TASK_TYPES)
        
        try:
            # Generate prompt
            prompt = generate_prompt(topic, task_type)
            
            # Generate golden response
            golden_response = generate_golden_response(prompt)
            
            sample = {
                "prompt": prompt,
                "golden_response": golden_response
            }
            
            dataset.append(sample)
            
            # Save incrementally every 10 samples
            if (i + 1) % 10 == 0:
                print(f"\n✓ Generated {i + 1}/{num_samples} samples")
                
        except Exception as e:
            print(f"\n✗ Error generating sample {i + 1}: {e}")
            continue
    
    return dataset


def main():
    args = parse_args()
    random.seed(args.seed)

    global client  # noqa: PLW0603 - keep existing function signatures
    global MODEL_NAME  # noqa: PLW0603
    client = make_client(args.base_url, args.api_key)
    MODEL_NAME = args.model

    print("=" * 60)
    print("Data Generation Script for Self-Distillation")
    print("=" * 60)
    print(f"vLLM Server: {args.base_url}")
    print(f"Model: {MODEL_NAME}")
    print()
    
    # Test connection
    print("Testing connection to vLLM server...")
    try:
        models = client.models.list()
        print(f"✓ Connected! Available models: {[m.id for m in models.data]}")
    except Exception as e:
        print(f"✗ Failed to connect to vLLM server: {e}")
        print(f"Make sure vLLM is running on {args.base_url}")
        return
    
    print()
    
    # Generate training data
    print("Generating training data...")
    train_data = generate_dataset(args.train_samples, desc="Training samples")
    
    print(f"\n✓ Generated {len(train_data)} training samples")
    
    # Generate evaluation data
    print("\nGenerating evaluation data...")
    eval_data = generate_dataset(args.eval_samples, desc="Evaluation samples")
    
    print(f"\n✓ Generated {len(eval_data)} evaluation samples")
    
    # Save training data
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    train_path = out_dir / "train_data.json"
    with open(train_path, "w", encoding="utf-8") as f:
        json.dump(train_data, f, indent=2, ensure_ascii=False)
    print(f"\n✓ Saved training data to {train_path}")
    
    # Save evaluation data
    eval_path = out_dir / "eval_data.json"
    with open(eval_path, "w", encoding="utf-8") as f:
        json.dump(eval_data, f, indent=2, ensure_ascii=False)
    print(f"✓ Saved evaluation data to {eval_path}")
    
    print("\n" + "=" * 60)
    print("Data Generation Complete!")
    print("=" * 60)
    print(f"Training samples: {len(train_data)}")
    print(f"Evaluation samples: {len(eval_data)}")
    print("\nYou can now run training with:")
    print("  python main.py --output_dir ./output")
    print("=" * 60)


if __name__ == "__main__":
    main()
