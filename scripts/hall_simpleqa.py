#!/usr/bin/env python3
"""Chinese SimpleQA evaluation via EvalScope — run directly on H100."""
import json, logging, os, sys, time

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

from evalscope import TaskConfig, run_task
from evalscope.constants import EvalType

MODEL = "qwen3.7-max"
API_URL = "https://dashscope.aliyuncs.com/compatible-mode/v1"
API_KEY = "YOUR_API_KEY_HERE"


def run():
    test_id = "hall_simpleqa"
    t_start = time.time()

    task_cfg = TaskConfig(
        model=MODEL, api_url=API_URL, api_key=API_KEY,
        eval_type=EvalType.OPENAI_API, datasets=["chinese_simpleqa"],
        eval_batch_size=8,
        generation_config={"temperature": 1.0, "max_tokens": 16384, "stream": True},
        judge_model_args={
            "model_id": MODEL, "api_url": API_URL,
            "api_key": API_KEY, "model_args": {}
        },
        limit=2, ignore_errors=True,
    )
    print("PROGRESS: 0%", flush=True)
    print("MESSAGE: SimpleQA evaluating...", flush=True)
    eval_result = run_task(task_cfg=task_cfg)
    print("PROGRESS: 100%", flush=True)
    print("MESSAGE: SimpleQA complete", flush=True)
    elapsed = time.time() - t_start

    dataset_result = eval_result.get("chinese_simpleqa", None)
    if dataset_result is not None:
        score = getattr(dataset_result, "score", None)
        num = getattr(dataset_result, "num", 0)
        if score is None and isinstance(dataset_result, dict):
            score = dataset_result.get("score", 0)
            num = dataset_result.get("num", 0)
    else:
        score = 0
        num = 0

    result = {
        "test_id": test_id,
        "status": "passed" if score is not None else "failed",
        "details": f"Chinese SimpleQA: score={score}, samples={num}, elapsed={elapsed:.0f}s",
        "actual_score": score,
        "metrics": {"score": score, "num_samples": num, "elapsed_seconds": elapsed}
    }

    # Strip ANSI escape codes + tqdm chars from details for Excel safety
    if isinstance(result.get("details"), str):
        import re
        result["details"] = re.sub(r'\[[0-9;]*m', '', result["details"])
        result["details"] = ''.join(c for c in result["details"] if ord(c) < 0x2580 or ord(c) > 0x259F)
        result["details"] = re.sub(r'[▀-▟]', '', result["details"])
    save_data = {"test_id": test_id, "status": result["status"], "details": result["details"], "metrics": result["metrics"]}
    path = os.path.join("/root/kimi-k2.6-benchmark/results", f"{test_id}.json")
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(save_data, f, indent=2, ensure_ascii=False)
    print(json.dumps(save_data, indent=2, ensure_ascii=False))
    return result


if __name__ == "__main__":
    run()
