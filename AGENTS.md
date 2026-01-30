## Working agreements (for coding agents)

### Quick commands
- Activate env: `source distillation/bin/activate`
- Verify env: `python scripts/check_env.py`
- Run training (example): `python3 main.py --output_dir ./output --vllm_server_base_url http://localhost:8000 ...`

### CLI conventions
- `main.py` uses Hugging Face `HfArgumentParser` (dataclass-driven). Extra positional args will error.
- Exception: the first positional argument may be the vLLM teacher server base URL (e.g. `http://localhost:8000` or `localhost:8000`), which is treated as `--vllm_server_base_url`.
- `--output_dir` is required.

### Repo hygiene
- Don’t commit generated artifacts: `output/`, `data/generated/`, `wandb/`, `distillation/` (see `.gitignore`).
- Keep changes minimal and in existing style; prefer `rg` for search.
