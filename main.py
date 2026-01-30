from __future__ import annotations

import os
import sys
from dataclasses import dataclass, field
from pathlib import Path
from string import Template
from typing import Optional, Tuple

import torch
from datasets import Dataset
from transformers import AutoModelForCausalLM, AutoTokenizer, HfArgumentParser

from distil_config import DistilConfig
from distil_trainer import DistilTrainer


@dataclass
class RunConfig:
    model_name_or_path: str = field(default="Qwen/Qwen3-0.6B", metadata={"help": "Student model name or path."})
    train_data_path: Optional[str] = field(
        default=None,
        metadata={
            "help": "Path to training JSON dataset. If omitted, defaults to "
            "`data/tooluse_data/train_data.json` when present, otherwise `data/tooluse_data/sample_train.json`."
        },
    )
    eval_data_path: Optional[str] = field(
        default=None, metadata={"help": "Optional path to eval JSON dataset (unused by default)."}
    )
    local_files_only: bool = field(
        default=False, metadata={"help": "If set, do not download model/tokenizer weights from Hugging Face Hub."}
    )
    trust_remote_code: bool = field(
        default=True, metadata={"help": "Allow models with custom code (required for many chat models)."}
    )
    torch_dtype: str = field(
        default="auto", metadata={"help": "Model dtype: auto|bf16|fp16|fp32."}
    )
    num_prompts_per_batch: int = field(
        default=32, metadata={"help": "Convenience alias for gradient_accumulation_steps (effective prompts per step)."}
    )
    dry_run: bool = field(default=False, metadata={"help": "Load everything and exit (sanity check)."})


def _cli_has_flag(flag: str) -> bool:
    return any(arg == flag or arg.startswith(f"{flag}=") for arg in sys.argv)


def _maybe_consume_positional_vllm_base_url() -> Optional[str]:
    """
    Allow `python main.py localhost:8000 ...` as a shorthand for `--vllm_server_base_url`.

    Hugging Face's `HfArgumentParser` does not support positional arguments; any extra token would
    otherwise raise an "arguments are not used" error.
    """
    if len(sys.argv) < 2:
        return None
    candidate = sys.argv[1]
    if candidate.startswith("-"):
        return None

    # Accept http(s) URLs, host:port, or [ipv6]:port.
    lowered = candidate.lower()
    looks_like_url = lowered.startswith("http://") or lowered.startswith("https://")
    looks_like_host_port = (
        "/" not in candidate
        and ":" in candidate
        and candidate.rsplit(":", 1)[-1].isdigit()
        and not lowered.startswith("file:")
    )
    if not (looks_like_url or looks_like_host_port):
        return None

    sys.argv.pop(1)
    if looks_like_url:
        return candidate
    return f"http://{candidate}"


def _configure_torch_for_cuda() -> None:
    if not torch.cuda.is_available():
        return
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    try:
        torch.set_float32_matmul_precision("high")
    except Exception:
        pass


def _resolve_dtype(dtype: str) -> torch.dtype:
    dtype = (dtype or "auto").lower()
    if dtype == "bf16":
        return torch.bfloat16
    if dtype == "fp16":
        return torch.float16
    if dtype in {"fp32", "float32"}:
        return torch.float32
    # auto
    if torch.cuda.is_available():
        # Prefer bf16 on modern NVIDIA GPUs; fall back to fp16.
        if getattr(torch.cuda, "is_bf16_supported", lambda: False)():
            return torch.bfloat16
        return torch.float16
    return torch.float32


def load_tooluse_dataset(train_path: str, seed: int = 42) -> Dataset:
    """Load and format a JSON dataset with keys: `prompt` (str) and `golden_response` (list[str])."""
    path = Path(train_path)
    if not path.exists():
        raise FileNotFoundError(
            f"Missing dataset at `{train_path}`. Generate it with `python generate_data.py` or "
            "pass `--train_data_path`."
        )

    dataset = Dataset.from_json(str(path))

    teacher_prompt_tmpl = Template(
        """
$orig_content

Example answer:
$output_text

Now answer the original question in your own words. Explain your reasoning clearly and provide code if needed.
""".strip()
    )

    def format_example(example):
        prompt_text = example["prompt"]
        golden = example.get("golden_response")
        if isinstance(golden, list):
            golden_text = "\n".join(golden)
        else:
            golden_text = str(golden) if golden is not None else ""

        return {
            "prompt": [{"role": "user", "content": prompt_text}],
            "teacher_prompt": [
                {"role": "user", "content": teacher_prompt_tmpl.substitute(orig_content=prompt_text, output_text=golden_text)}
            ],
        }

    dataset = dataset.map(format_example, remove_columns=dataset.column_names)
    dataset = dataset.shuffle(seed=seed)
    return dataset


def main() -> None:
    positional_base_url = _maybe_consume_positional_vllm_base_url()

    parser = HfArgumentParser((RunConfig, DistilConfig))
    run_cfg, train_cfg = parser.parse_args_into_dataclasses()

    _configure_torch_for_cuda()

    # Provide safe, teacher-server defaults unless the user explicitly sets them.
    if not _cli_has_flag("--use_vllm") and not _cli_has_flag("--no_use_vllm"):
        train_cfg.use_vllm = True
    if not _cli_has_flag("--vllm_mode"):
        train_cfg.vllm_mode = "server"
    if not _cli_has_flag("--generate_from_teacher") and not _cli_has_flag("--no_generate_from_teacher"):
        train_cfg.generate_from_teacher = True
    if not _cli_has_flag("--report_to"):
        train_cfg.report_to = "none"
    if not _cli_has_flag("--per_device_train_batch_size"):
        train_cfg.per_device_train_batch_size = 1
    if not _cli_has_flag("--gradient_accumulation_steps"):
        train_cfg.gradient_accumulation_steps = int(run_cfg.num_prompts_per_batch)
    if not _cli_has_flag("--max_prompt_length"):
        train_cfg.max_prompt_length = 1024
    if not _cli_has_flag("--max_completion_length"):
        train_cfg.max_completion_length = 1024
    if not _cli_has_flag("--save_steps"):
        train_cfg.save_steps = 100
    if not _cli_has_flag("--logging_steps"):
        train_cfg.logging_steps = 1
    if not _cli_has_flag("--sync_ref_model") and not _cli_has_flag("--no_sync_ref_model"):
        train_cfg.sync_ref_model = False
    if not _cli_has_flag("--vllm_importance_sampling_correction") and not _cli_has_flag("--no_vllm_importance_sampling_correction"):
        train_cfg.vllm_importance_sampling_correction = False

    if train_cfg.use_vllm and train_cfg.vllm_mode == "server":
        if positional_base_url is not None and not _cli_has_flag("--vllm_server_base_url"):
            train_cfg.vllm_server_base_url = positional_base_url

        base_url_set = train_cfg.vllm_server_base_url is not None

        if not base_url_set and not _cli_has_flag("--vllm_server_host"):
            train_cfg.vllm_server_host = "localhost"
        if not base_url_set and not _cli_has_flag("--vllm_server_port"):
            train_cfg.vllm_server_port = 8000
        if not _cli_has_flag("--vllm_server_model"):
            train_cfg.vllm_server_model = "glm-4.7"

    # Auto bf16/fp16 selection unless explicitly set.
    if not _cli_has_flag("--bf16") and not _cli_has_flag("--no_bf16") and not _cli_has_flag("--fp16") and not _cli_has_flag("--no_fp16"):
        if torch.cuda.is_available() and getattr(torch.cuda, "is_bf16_supported", lambda: False)():
            train_cfg.bf16 = True
            train_cfg.fp16 = False
        elif torch.cuda.is_available():
            train_cfg.bf16 = False
            train_cfg.fp16 = True

    if not train_cfg.output_dir:
        raise ValueError("`--output_dir` is required (it comes from DistilConfig / TrainingArguments).")

    # Make output_dir deterministic and create parent folder early.
    Path(train_cfg.output_dir).mkdir(parents=True, exist_ok=True)

    if torch.cuda.is_available():
        gpu_name = torch.cuda.get_device_name(0)
        cc = torch.cuda.get_device_capability(0)
        print(f"CUDA: {torch.version.cuda} | GPU0: {gpu_name} (cc={cc[0]}.{cc[1]})")
    else:
        print("CUDA not available; training will run on CPU (not recommended).")

    model_dtype = _resolve_dtype(run_cfg.torch_dtype)
    try:
        model = AutoModelForCausalLM.from_pretrained(
            run_cfg.model_name_or_path,
            torch_dtype=model_dtype,
            trust_remote_code=run_cfg.trust_remote_code,
            low_cpu_mem_usage=True,
            local_files_only=run_cfg.local_files_only,
        )
    except Exception as exc:
        raise RuntimeError(
            "Failed to load the student model. If you're offline (or running in a restricted environment), "
            "pre-download the model/tokenizer or re-run with `--local_files_only`. "
            "Example: `python3 main.py --local_files_only ...`"
        ) from exc
    model.config.use_cache = False

    try:
        tokenizer = AutoTokenizer.from_pretrained(
            run_cfg.model_name_or_path,
            trust_remote_code=run_cfg.trust_remote_code,
            local_files_only=run_cfg.local_files_only,
            padding_side="left",
            truncation_side="left",
        )
    except Exception as exc:
        raise RuntimeError(
            "Failed to load the tokenizer. If you're offline, pre-download it or re-run with `--local_files_only`."
        ) from exc
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    train_path = run_cfg.train_data_path
    if not train_path:
        preferred = Path("data/tooluse_data/train_data.json")
        fallback = Path("data/tooluse_data/sample_train.json")
        train_path = str(preferred if preferred.exists() else fallback)

    dataset = load_tooluse_dataset(train_path, seed=train_cfg.seed)

    trainer = DistilTrainer(
        model=model,
        ref_model=None,  # beta=0 defaults to no reference model
        args=train_cfg,
        train_dataset=dataset,
        processing_class=tokenizer,
    )

    if run_cfg.dry_run:
        print("Dry run complete.")
        return

    trainer.train()


if __name__ == "__main__":
    # Avoid tokenizer parallelism warnings; harmless but noisy.
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    main()
