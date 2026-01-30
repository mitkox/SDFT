# On-Policy Self-Distillation with External Teacher

This is TRL-based code for reproducing the paper "Self-Distillation Enables Continual Learning", modified to support external vLLM teacher models for efficient distillation.

## Key Features

- **External Teacher Support**: Uses a separate vLLM server running a large teacher model (e.g., GLM-4.7 30B MoE)
- **Efficient Student Training**: Student model (e.g., Qwen3-0.6B) runs on GPU for fast training
- **Distillation from Demonstrations**: Learn from teacher-generated examples with in-context learning
- **Memory Optimized**: Both models can run on a single GPU by using external vLLM server mode

## Abstract

Continual learning, enabling models to acquire new skills and knowledge without degrading existing capabilities, remains a fundamental challenge for foundation models. While on-policy reinforcement learning can reduce forgetting, it requires explicit reward functions that are often unavailable. Learning from expert demonstrations, the primary alternative, is dominated by supervised fine-tuning (SFT), which is inherently off-policy. We introduce **Self-Distillation Fine-Tuning (SDFT)**, a simple method that enables on-policy learning directly from demonstrations. SDFT leverages in-context learning by using a demonstration-conditioned model as its own teacher, generating on-policy training signals that preserve prior capabilities while acquiring new skills. Across skill learning and knowledge acquisition tasks, SDFT consistently outperforms SFT, achieving higher new-task accuracy while substantially reducing catastrophic forgetting. In sequential learning experiments, SDFT enables a single model to accumulate multiple skills over time without performance regression, establishing on-policy distillation as a practical path to continual learning from demonstrations.

## Setup

### 1. Clone the repository

```bash
git clone https://github.com/mitkox/SDFT.git
cd SDFT
```

### 2. Set up a virtual environment

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

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

**Important**: If you have a newer NVIDIA GPU (e.g., GB10), you may need to upgrade PyTorch:

```bash
pip install --upgrade torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124
```

### 4. Set up the Teacher Model (vLLM Server)

Start a vLLM server with your teacher model on port 8000:

```bash
vllm serve THUDM/glm-4-9b-chat --port 8000
```

Or for a larger model:

```bash
vllm serve <your-teacher-model> --port 8000 --tensor-parallel-size 1
```

The teacher model generates high-quality training examples that the student model learns from.

### 5. Prepare Your Data

Place your training data in `data/tooluse_data/`:
- `train_data.json` - Training examples
- `eval_data.json` - Evaluation examples (optional)

Data format:
```json
[
  {
    "prompt": "Your task prompt",
    "golden_response": ["Expected response"]
  }
]
```

### 6. Run Training

```bash
python main.py \
  --model_name Qwen/Qwen3-0.6B \
  --output_dir ./output \
  --learning_rate 2e-5 \
  --num_train_epochs 1 \
  --num_prompts_per_batch 32
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
- **Teacher**: External GLM-4.7 model on port 8000
- **Student**: Qwen3-0.6B loaded on GPU
- **Batch Size**: 1 per device with gradient accumulation
- **Max Lengths**: 1024 tokens for both prompt and completion

## Arguments

| Argument | Default | Description |
|----------|---------|-------------|
| `--model_name` | `Qwen/Qwen3-0.6B` | Student model path |
| `--output_dir` | - | Output directory for checkpoints |
| `--learning_rate` | `2e-5` | Learning rate |
| `--num_train_epochs` | `1` | Number of training epochs |
| `--num_prompts_per_batch` | `32` | Prompts per batch (gradient accumulation) |
| `--ref_model_mixup_alpha` | `0.01` | Reference model mixup alpha |
| `--seed` | `42` | Random seed |

## Troubleshooting

### CUDA Compatibility Error
If you see "no kernel image is available for execution on the device", upgrade PyTorch:
```bash
pip install --upgrade torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124
```

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
