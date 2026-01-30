# Self-Distillation (External Teacher via vLLM)

This repo provides a TRL-based training loop for on-policy self-distillation using an external teacher served via vLLM (OpenAI-compatible HTTP API).

## Key Features

- **External Teacher Support**: Uses a separate vLLM server running a large teacher model (e.g., GLM-4.7 30B MoE)
- **Efficient Student Training**: Student model (e.g., Qwen3-0.6B) runs on GPU for fast training
- **Distillation from Demonstrations**: Learn from teacher-generated examples with in-context learning
- **Memory Optimized**: Both models can run on a single GPU by using external vLLM server mode

## Abstract

Continual learning, enabling models to acquire new skills and knowledge without degrading existing capabilities, remains a fundamental challenge for foundation models. While on-policy reinforcement learning can reduce forgetting, it requires explicit reward functions that are often unavailable. Learning from expert demonstrations, the primary alternative, is dominated by supervised fine-tuning (SFT), which is inherently off-policy. We introduce **Self-Distillation Fine-Tuning (SDFT)**, a simple method that enables on-policy learning directly from demonstrations. SDFT leverages in-context learning by using a demonstration-conditioned model as its own teacher, generating on-policy training signals that preserve prior capabilities while acquiring new skills. Across skill learning and knowledge acquisition tasks, SDFT consistently outperforms SFT, achieving higher new-task accuracy while substantially reducing catastrophic forgetting. In sequential learning experiments, SDFT enables a single model to accumulate multiple skills over time without performance regression, establishing on-policy distillation as a practical path to continual learning from demonstrations.

## Setup

### 1) Clone the repository

```bash
git clone <YOUR_GITHUB_REPO_URL>.git
cd Self-Distillation
```

### 2) Set up a virtual environment

Using **venv**:

```bash
python3.12 -m venv distillation
source distillation/bin/activate
```

Using **conda**:

```bash
conda create -n distillation python=3.12
conda activate distillation
```

### 3) Install dependencies

Install PyTorch first, then Python dependencies.

**For NVIDIA GB10 (CUDA 13.1):**

```bash
# Install PyTorch nightly with CUDA 13.x support first (CUDA 13.1 driver is fine).
# If `cu131` wheels are not available yet, try `cu130`.
pip uninstall -y torch torchvision torchaudio
pip install --pre --upgrade --no-cache-dir \
  torch torchvision torchaudio \
  --index-url https://download.pytorch.org/whl/nightly/cu131
  # fallback:
  # --index-url https://download.pytorch.org/whl/nightly/cu130

# Then install other dependencies
pip install -r requirements.txt
```

**For older GPUs (CUDA 12.4 and below):**

```bash
pip install -r requirements.txt
pip install --upgrade torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124
```

### 4) Set up the Teacher (vLLM server)

Start a vLLM server with your teacher model on port 8000:

```bash
vllm serve <your-teacher-model> --port 8000 --served-model-name glm-4.7
```

If you are using tensor parallelism:

```bash
vllm serve <your-teacher-model> --port 8000 --served-model-name glm-4.7 --tensor-parallel-size 1
```

The teacher model generates high-quality training examples that the student model learns from.

Verify the server:

```bash
curl http://localhost:8000/v1/models
```

Tip (continuous batching): If you run vLLM with continuous batching, you can often increase training throughput by
setting `--vllm_server_max_concurrency` to a value like `4-16` so the trainer submits multiple parallel requests.

Verify your Python/PyTorch/CUDA stack:

```bash
python scripts/check_env.py
```

### 5) Prepare data

This repo includes an example dataset in `data/tooluse_data/`:
- `data/tooluse_data/train_data.json` - training examples
- `data/tooluse_data/eval_data.json` - evaluation examples
- `data/tooluse_data/sample_train.json` - tiny smoke-test dataset

Data format:
```json
[
  {
    "prompt": "Your task prompt",
    "golden_response": ["Expected response"]
  }
]
```

Generate a dataset with the provided script (writes to `data/generated/tooluse_data/` by default):

```bash
python3 generate_data.py --base_url http://localhost:8000/v1 --model glm-4.7 --train_samples 100 --eval_samples 20
```

### 6) Run training

```bash
python3 main.py \
  --output_dir ./output \
  --model_name_or_path Qwen/Qwen3-0.6B \
  --train_data_path data/tooluse_data/train_data.json \
  --learning_rate 2e-5 \
  --num_train_epochs 1 \
  --num_prompts_per_batch 32 \
  --vllm_server_base_url http://localhost:8000 \
  --vllm_server_model glm-4.7 \
  --vllm_server_max_concurrency 8
```

Shorthand: you can also pass the teacher server base URL as the first positional argument, e.g.:

```bash
python3 main.py http://localhost:8000 --output_dir ./output --vllm_server_model glm-4.7
```

If your machine cannot download from Hugging Face (offline), add `--local_files_only` and make sure the model is cached.

Dataset selection behavior:
- If you **don’t** pass `--train_data_path`, `main.py` will use `data/tooluse_data/train_data.json` if it exists, otherwise it falls back to `data/tooluse_data/sample_train.json`.
- To train on newly generated data, point `--train_data_path` at `data/generated/tooluse_data/train_data.json`.

Sanity check without training:

```bash
python3 main.py --output_dir ./output --dry_run
```

## Architecture

```
┌─────────────────────────┐
│  Teacher Model (vLLM)   │  ← Large model (e.g., GLM-4.7 30B)
│  Port 8000 (External)   │     Generates training examples
└───────────┬─────────────┘
            │ HTTP API
            ▼
┌─────────────────────────┐
│   Training Loop (GPU)   │  ← Student model (e.g., Qwen3-0.6B)
│   Student Model         │     Learns from teacher outputs
└─────────────────────────┘
```

## Configuration

The training uses the following key configurations:

- **vLLM Mode**: `server` - connects to external vLLM server
- **Teacher**: External teacher model served as `glm-4.7` on port 8000
- **Student**: Qwen3-0.6B loaded on GPU
- **Batch Size**: 1 per device with gradient accumulation
- **Max Lengths**: 1024 tokens for both prompt and completion

## Arguments

| Argument | Default | Description |
|----------|---------|-------------|
| `--model_name_or_path` | `Qwen/Qwen3-0.6B` | Student model name/path |
| `--train_data_path` | auto | Training JSON (auto-picks `data/tooluse_data/train_data.json` if present) |
| `--output_dir` | - | Output directory |
| `--num_prompts_per_batch` | `32` | Convenience alias for `--gradient_accumulation_steps` |
| `--vllm_server_base_url` | - | Teacher server base URL (e.g., `http://localhost:8000`) |
| `--vllm_server_model` | `glm-4.7` | Served teacher model name |
| `--vllm_server_max_concurrency` | `1` | Concurrent HTTP requests to vLLM (useful with continuous batching) |
| `--dry_run` | `False` | Load everything and exit |

## Troubleshooting

### CUDA Compatibility Error (GB10 GPU)
If you see "no kernel image is available for execution on the device" with NVIDIA GB10 (RTX 50 series), you need PyTorch nightly:

```bash
pip uninstall -y torch torchvision torchaudio
pip install --pre --upgrade --no-cache-dir \
  torch torchvision torchaudio \
  --index-url https://download.pytorch.org/whl/nightly/cu130
```

**Note**: You may see a dependency conflict warning about `cuda-python` and `cuda-bindings` versions. This is expected and won't prevent the code from working.

### vLLM Server Connection Error
Ensure your vLLM server is running and accessible:
```bash
curl http://localhost:8000/v1/models
```

### Out of Memory
If you encounter OOM errors:
- Reduce `per_device_train_batch_size` (currently 1)
- Reduce `max_prompt_length` or `max_completion_length`
- Use gradient checkpointing (already enabled)

## Citation

If you use this code, please cite the original paper:

```bibtex
@article{sdft2025,
  title={Self-Distillation Enables Continual Learning},
  author={...},
  year={2025}
}
```
