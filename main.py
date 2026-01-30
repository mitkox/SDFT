from distil_trainer import DistilTrainer
from distil_config import DistilConfig
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
from datasets import Dataset, load_dataset, load_from_disk
from string import Template
import argparse
import torch.distributed as dist

def parse_args():
    parser = argparse.ArgumentParser(description="Distil Trainer")
    parser.add_argument("--learning_rate", type=float, default=2e-5, help="Learning rate")
    parser.add_argument("--num_train_epochs", type=int, default=1, help="Number of training epochs")
    parser.add_argument("--num_prompts_per_batch", type=int, default=32, help="Number of prompts per batch")
    parser.add_argument("--ref_model_mixup_alpha", type=float, default=0.01, help="Reference model mixup alpha")
    parser.add_argument("--output_dir", type=str, help="Output directory")
    parser.add_argument("--model_name", type=str, default="Qwen/Qwen3-0.6B", help="Model name")
    parser.add_argument("--seed", type=int, default=42, help="Seed")
    return parser.parse_args()

def load_tooluse_dataset(seed=42) -> Dataset:
    """Load and prepare tooluse dataset with formatted prompts."""
    train_path = 'data/tooluse_data/train_data.json'
    test_path = 'data/tooluse_data/eval_data.json'
    train_dataset = Dataset.from_json(train_path)
    test_dataset = Dataset.from_json(test_path)

    def format_example(example):

        teacher_prompt = Template("""
$orig_content

This is an example for a response to the question:
$output_text

Now answer with a response of your own, including the thinking process.
""")

        return {
            "prompt": [{"role": "user", "content": example['prompt']}],
            "teacher_prompt": [{"role": "user", "content": teacher_prompt.substitute(orig_content=example['prompt'], output_text='\n'.join(example['golden_response']))}],
        }
    
    train_dataset = train_dataset.map(format_example, remove_columns=train_dataset.column_names)
    train_dataset = train_dataset.shuffle(seed=seed)
    return train_dataset, None


if __name__ == "__main__":
    args = parse_args()
    print("Loading model with memory optimization...")
    # GPU enabled with PyTorch nightly (CUDA 13.0) for GB10 support
    model = AutoModelForCausalLM.from_pretrained(
        args.model_name,
        torch_dtype=torch.bfloat16,
        trust_remote_code=True,
        device_map="auto",  # GPU with PyTorch nightly cu130
        low_cpu_mem_usage=True,
    )
    # Load a local "teacher" model for computing logits (needed for distillation loss)
    # Since we don't have GLM-4.7 locally, use the student model as teacher
    # External vLLM server generates the completions, but we need local model for logit computation
    print("Loading teacher model (copy of student for logit computation)...")
    teacher_model = AutoModelForCausalLM.from_pretrained(
        args.model_name,
        torch_dtype=torch.bfloat16,
        trust_remote_code=True,
        device_map="auto",
        low_cpu_mem_usage=True,
    )
    tokenizer = AutoTokenizer.from_pretrained(
        args.model_name, 
        trust_remote_code=True,
        local_files_only=True  # Use cached files only, don't contact HuggingFace
    )
    dataset, _ = load_tooluse_dataset(args.seed)  # Unpack tuple

    config = DistilConfig(
        seed=args.seed,
        use_vllm = True,
        vllm_mode="server",  # Use external GLM-4.7 teacher server
        vllm_server_host="localhost",
        vllm_server_port=8000,
        # generate_from_teacher not needed - server mode uses external vLLM model automatically
        vllm_tensor_parallel_size=1, 
        vllm_gpu_memory_utilization=0.3,
        vllm_enable_sleep_mode=False,  # Not applicable for external server 
        learning_rate = args.learning_rate,
        warmup_ratio = 0.1,
        lr_scheduler_type = "cosine",
        logging_steps = 1,
        bf16 = True,  # GPU supports bfloat16
        fp16 = False,
        per_device_train_batch_size = 1,
        gradient_accumulation_steps = args.num_prompts_per_batch,
        max_prompt_length = 1024,
        max_completion_length = 1024,
        num_train_epochs = args.num_train_epochs,
        max_steps = 100,  # Set max steps since dataset may not have length
        save_steps = 100,
        max_grad_norm = 1,
        report_to = "none",  # Disable wandb to avoid login requirements
        output_dir = args.output_dir,
        log_completions = False, # True for debugging
        sync_ref_model = False,  # Can't sync to external vLLM server
        ref_model_sync_steps = 1,
        ref_model_mixup_alpha = args.ref_model_mixup_alpha,
        vllm_importance_sampling_correction = False,  # Disabled for external server
        num_loss_tokens_to_skip = 3,
        alpha = 0.0,  # No KL distillation - pure imitation learning from teacher generations
        beta = 0.0,  # No KL penalty with reference model
        full_logit_distillation = False,  # Don't compute teacher logits (teacher is external)
    )
    trainer = DistilTrainer(
        model=model,
        ref_model=teacher_model,  # Local teacher for logit computation (same as student initially)
        args=config,
        train_dataset=dataset,
        processing_class=tokenizer,
    )
    trainer.train()
