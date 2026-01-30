# Tooluse dataset

`main.py` expects a JSON array with items like:

```json
[
  {
    "prompt": "Your question/task",
    "golden_response": ["Step 1 ...", "Step 2 ..."]
  }
]
```

This repo includes:
- `train_data.json` / `eval_data.json`: original dataset files from the upstream repo.
- `sample_train.json`: a tiny dataset to validate the training pipeline quickly.

To generate additional data without overwriting the bundled dataset, use:

```bash
python generate_data.py --base_url http://localhost:8000/v1 --model glm-4.7 --train_samples 100 --eval_samples 20 --out_dir data/generated/tooluse_data
```
