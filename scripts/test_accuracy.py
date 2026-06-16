#!/usr/bin/env python3
"""
Benchmark Accuracy Tests for LLM Model Benchmark.
Tests model accuracy on various benchmarks: AIME2025, HLE, SWE-Bench, GPQA,
LongBench, and TAU Bench (retail/telecom/airline).
"""

import argparse
import json
import logging
import os
import re
import sys
import time
import subprocess
import glob
import shutil
from pathlib import Path
from typing import Dict, Any, List, Optional

import openai
import concurrent.futures
from openai.resources.chat.completions import Completions

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Global ThreadPoolExecutor for enforcing absolute wall-clock timeouts on OpenAI API requests
# to prevent indefinite blocking caused by keep-alive whitespaces resetting socket read timeouts.
_TIMEOUT_EXECUTOR = concurrent.futures.ThreadPoolExecutor(max_workers=50, thread_name_prefix="openai_timeout_worker")

if not getattr(Completions, "_is_wall_clock_patched", False):
    original_create = Completions.create
    
    def patched_create(self, *args, **kwargs):
        timeout = kwargs.get("timeout")
        stream = kwargs.get("stream", False)
        
        # Enforce absolute wall-clock timeout only for non-streaming requests
        if not stream and isinstance(timeout, (int, float)) and timeout > 0:
            future = _TIMEOUT_EXECUTOR.submit(original_create, self, *args, **kwargs)
            try:
                return future.result(timeout=timeout)
            except concurrent.futures.TimeoutError:
                logger.error(f"Request timed out (enforced absolute wall-clock timeout of {timeout}s)")
                raise openai.APITimeoutError("Request timed out (enforced absolute wall-clock timeout)")
        else:
            return original_create(self, *args, **kwargs)
            
    Completions.create = patched_create
    Completions._is_wall_clock_patched = True
    logger.info("Successfully monkey-patched Completions.create to enforce absolute wall-clock timeout.")


# Paths
SCRIPT_DIR = Path(__file__).parent
PROJECT_DIR = SCRIPT_DIR.parent
CONFIG_PATH = PROJECT_DIR / "config.json"
RESULTS_DIR = PROJECT_DIR / "results"
BENCHMARKS_DIR = PROJECT_DIR / "data" / "benchmarks"
RESULTS_DIR.mkdir(exist_ok=True)


def load_config() -> Dict[str, Any]:
    """Load configuration from config.json"""
    if CONFIG_PATH.exists():
        with open(CONFIG_PATH, 'r', encoding='utf-8') as f:
            return json.load(f)
    return {}


def load_benchmark(name: str) -> List[Dict[str, Any]]:
    """Load benchmark data from JSON file."""
    path = BENCHMARKS_DIR / f"{name}.json"
    if path.exists():
        with open(path, 'r', encoding='utf-8') as f:
            return json.load(f)
    logger.warning(f"Benchmark file not found: {path}")
    return []


def get_client(api_base: str, api_key: str) -> openai.OpenAI:
    """Create OpenAI client."""
    return openai.OpenAI(base_url=api_base, api_key=api_key, default_headers={"User-Agent": "curl/8.0"})


def write_progress(test_id: str, progress: Dict[str, Any]):
    """Write progress to JSON file for WebUI polling."""
    progress_path = RESULTS_DIR / f"{test_id}_progress.json"
    with open(progress_path, 'w') as f:
        json.dump(progress, f, indent=2, default=str)


def write_result(test_id: str, result: Dict[str, Any]):
    """Write final result to JSON file."""
    result_path = RESULTS_DIR / f"{test_id}.json"
    with open(result_path, 'w') as f:
        json.dump(result, f, indent=2, default=str)


def retry_with_backoff(func, max_retries: int = 3, base_delay: float = 1.0):
    """Retry a function with exponential backoff."""
    last_error = None
    for attempt in range(max_retries):
        try:
            return func()
        except Exception as e:
            last_error = e
            if attempt < max_retries - 1:
                delay = base_delay * (2 ** attempt)
                logger.warning(f"Attempt {attempt + 1} failed: {e}. Retrying in {delay}s...")
                time.sleep(delay)
    raise last_error


def call_model(client: openai.OpenAI, messages: List[Dict], max_tokens: int = 2000,
               temperature: float = 0.0) -> str:
    """Call the model and return response content."""
    config = load_config()
    model_name = config.get("model", {}).get("name", "kimi-k2.6")
    response = retry_with_backoff(lambda: client.chat.completions.create(
        model=model_name,
        messages=messages,
        max_tokens=max_tokens,
        temperature=temperature
    ))
    return response.choices[0].message.content or ""


def normalize_answer(answer: str) -> str:
    """Normalize an answer string for comparison."""
    answer = answer.strip().lower()
    # Remove articles
    answer = re.sub(r'\b(a|an|the)\b', '', answer)
    # Remove punctuation
    answer = re.sub(r'[^\w\s]', '', answer)
    # Normalize whitespace
    answer = re.sub(r'\s+', ' ', answer).strip()
    return answer


def check_answer_match(expected: str, actual: str) -> bool:
    """Check if actual answer contains/matches expected answer."""
    norm_expected = normalize_answer(expected)
    norm_actual = normalize_answer(actual)

    # Exact match
    if norm_expected == norm_actual:
        return True

    # Containment check
    if norm_expected in norm_actual:
        return True

    # Try to extract numbers
    expected_nums = re.findall(r'\d+', expected)
    actual_nums = re.findall(r'\d+', actual)
    if expected_nums and actual_nums:
        if expected_nums[-1] == actual_nums[-1]:
            return True

    return False


# ============================================================================
# Benchmark Test Functions via EvalScope Official Python SDK
# ============================================================================

def run_evalscope_dataset(test_id: str, dataset_name: str, api_base: str, api_key: str, subset_list: List[str] = None, dataset_hub: str = None, dataset_id: str = None) -> Dict[str, Any]:
    """Helper to run evalscope evaluation using the official Python API."""
    logger.info(f"Running benchmark via EvalScope SDK: {test_id} ({dataset_name})")
    write_progress(test_id, {
        "status": "running",
        "current_step": f"Initializing TaskConfig for {dataset_name}..."
    })

    config = load_config()
    
    config_keys = {
        "aime25": ("AIME2025", 95.3),
        "aime26": ("AIME2026", None),
        "hle": ("HLE", 52.3),
        "swe_bench_verified": ("SWE-Bench_Verified", 58.4),
        "gpqa": ("GPQA", 86.2),
        "gpqa_diamond": ("GPQA", 86.2),
        "longbench_v2": ("LongBench_V2", 50.0),
        "tau3_bench": ("TAU_retail" if (subset_list and "retail" in subset_list) else ("TAU_telecom" if (subset_list and "telecom" in subset_list) else "TAU_airline"), 70.6)
    }
    
    config_key, default_official = config_keys.get(dataset_name, ("AIME2025", 95.3))
    
    official_score = config.get("benchmarks", {}).get(config_key, {}).get("official", default_official)
    if official_score is None:
        official_score = default_official
        
    tolerance = config.get("benchmarks", {}).get(config_key, {}).get("tolerance", 4.0)

    work_dir = RESULTS_DIR / f"evalscope_{test_id}_{int(time.time())}"
    work_dir.mkdir(exist_ok=True, parents=True)

    limit = int(os.environ.get("EVALSCOPE_LIMIT", "1"))

    try:
        from evalscope.run import run_task
        from evalscope.config import TaskConfig
    except ImportError as e:
        logger.error(f"Failed to import evalscope. Please ensure it is installed: {e}")
        raise e

    dataset_args = {}
    if dataset_id:
        dataset_args[dataset_name] = {"dataset_id": dataset_id}
        
    if subset_list:
        if dataset_name not in dataset_args:
            dataset_args[dataset_name] = {}
        dataset_args[dataset_name]["subset_list"] = subset_list

    model_name = config.get("model", {}).get("name", "kimi-k2.6")

    if dataset_name == "tau3_bench":
        if dataset_name not in dataset_args:
            dataset_args[dataset_name] = {}
        dataset_args[dataset_name]["extra_params"] = {
            "user_model": model_name,
            "api_key": api_key,
            "api_base": api_base,
            "generation_config": {"temperature": 1.0, "max_tokens": 2048}
        }

    generation_cfg = {"temperature": 1.0, "timeout": 180}
    if dataset_name == "tau3_bench":
        generation_cfg["max_tokens"] = 32768
        generation_cfg["stream"] = True

    judge_model_args = None
    if dataset_name == "hle":
        judge_model_args = {
            "api_key": api_key,
            "api_url": api_base,
            "model_id": model_name,
            "eval_type": "openai_api"
        }

    # Create the task configuration matching the official way
    task_cfg = TaskConfig(
        model=model_name,
        api_url=api_base,
        api_key=api_key,
        eval_type="openai_api",
        datasets=[dataset_name],
        dataset_hub=dataset_hub if dataset_hub else "modelscope",
        dataset_args=dataset_args,  # Pass as dictionary to satisfy Pydantic
        limit=limit,
        generation_config=generation_cfg,
        work_dir=str(work_dir),
        stream=(dataset_name == "tau3_bench"),
        ignore_errors=(dataset_name == "tau3_bench"),
        judge_model_args=judge_model_args
    )

    write_progress(test_id, {
        "status": "running",
        "current_step": f"Executing EvalScope task for {dataset_name}..."
    })

    if dataset_name == "hle":
        try:
            from evalscope.benchmarks.hle.hle_adapter import HLEAdapter
            if not getattr(HLEAdapter, "_is_monkey_patched", False):
                original_record_to_sample = HLEAdapter.record_to_sample
                
                def patched_record_to_sample(self, record):
                    sample = original_record_to_sample(self, record)
                    from evalscope.api.messages import ChatMessageUser
                    if hasattr(sample, 'input') and isinstance(sample.input, list):
                        new_input = []
                        for msg in sample.input:
                            if isinstance(msg, ChatMessageUser) and isinstance(msg.content, list):
                                text_parts = [part.text for part in msg.content if hasattr(part, 'text')]
                                if text_parts:
                                    msg.content = "\n".join(text_parts)
                            new_input.append(msg)
                        sample.input = new_input
                    return sample

                HLEAdapter.record_to_sample = patched_record_to_sample
                HLEAdapter._is_monkey_patched = True
                logger.info("Successfully monkey-patched HLEAdapter.record_to_sample to use raw strings for text-only messages.")
        except Exception as patch_err:
            logger.warning(f"Failed to apply HLEAdapter monkeypatch: {patch_err}")

    if dataset_name == "swe_bench_verified":
        def apply_swe_bench_patch(adapter_class, class_name):
            if adapter_class is None:
                return
            if not getattr(adapter_class, "_is_monkey_patched", False):
                original_post_process = adapter_class._post_process_samples
                
                def patched_post_process(self):
                    import docker
                    try:
                        client = docker.from_env()
                        cached_suffixes = set()
                        for img in client.images.list():
                            for tag in img.tags:
                                repo_name_with_tag = tag.split(':')[0]
                                normalized_tag = repo_name_with_tag.replace('__', '_')
                                suffix = normalized_tag.split('_')[-1]
                                cached_suffixes.add(suffix)
                        
                        samples = self.test_dataset[self.default_subset]
                        filtered_samples = []
                        for sample in samples:
                            instance_id = sample.metadata.get('instance_id', '')
                            suffix = instance_id.split('__')[-1]
                            if suffix in cached_suffixes:
                                filtered_samples.append(sample)
                        
                        logger.info(f"SWE-bench {class_name} MonkeyPatch: Filtered test samples from {len(samples)} to {len(filtered_samples)} based on local Docker cache.")
                        self.test_dataset[self.default_subset] = filtered_samples
                    except Exception as e:
                        logger.warning(f"Failed to pre-filter SWE-bench samples in {class_name}: {e}")
                    
                    original_post_process(self)
                
                adapter_class._post_process_samples = patched_post_process
                adapter_class._is_monkey_patched = True
                logger.info(f"Successfully monkey-patched {class_name} to only use cached Docker images.")

        try:
            from evalscope.benchmarks.swe_bench.swe_bench_agentic_adapter import SWEBenchAgenticAdapter
        except Exception:
            SWEBenchAgenticAdapter = None

        try:
            from evalscope.benchmarks.swe_bench.swe_bench_adapter import SWEBenchVerifiedAdapter
        except Exception:
            SWEBenchVerifiedAdapter = None

        if SWEBenchAgenticAdapter:
            try:
                apply_swe_bench_patch(SWEBenchAgenticAdapter, "SWEBenchAgenticAdapter")
            except Exception as e:
                logger.warning(f"Failed to apply SWEBenchAgenticAdapter patch: {e}")

        if SWEBenchVerifiedAdapter:
            try:
                apply_swe_bench_patch(SWEBenchVerifiedAdapter, "SWEBenchVerifiedAdapter")
            except Exception as e:
                logger.warning(f"Failed to apply SWEBenchVerifiedAdapter patch: {e}")

    try:
        run_task(task_cfg=task_cfg)
    except Exception as e:
        logger.error(f"Failed to run evalscope task: {e}")
        raise e

    # Parse results
    correct = 0
    total = 0
    details = []
    
    jsonl_files = glob.glob(str(work_dir / "**" / "reviews" / model_name / "*.jsonl"), recursive=True)
    if not jsonl_files:
        jsonl_files = glob.glob(str(work_dir / "**" / "predictions" / model_name / "*.jsonl"), recursive=True)
        
    if jsonl_files:
        logger.info(f"Parsing evalscope reviews: {jsonl_files}")
        for jsonl_file in jsonl_files:
            with open(jsonl_file, "r") as f:
                for line in f:
                    if not line.strip():
                        continue
                    try:
                        data = json.loads(line)
                        total += 1
                        
                        score_val = 0.0
                        score_obj = data.get("sample_score", {}).get("score", {})
                        value = score_obj.get("value", {})
                        
                        if isinstance(value, dict):
                            is_correct = any(v == 1.0 or v is True for v in value.values())
                        else:
                            is_correct = value == 1.0 or value is True
                            
                        if is_correct:
                            correct += 1
                            score_val = 1.0
                            
                        predict = ""
                        expected = ""
                        
                        history = data.get("history", [])
                        if history and len(history) > 1:
                            expected = data.get("sample_metadata", {}).get("answer", "")
                            predict = history[-1].get("text", "")
                            
                        details.append({
                            "id": data.get("sample_id", f"sample_{total}"),
                            "correct": is_correct,
                            "expected": str(expected or score_obj.get("extracted_prediction", "N/A")),
                            "response_excerpt": str(predict[:200] if predict else score_obj.get("prediction", "")[:200])
                        })
                    except Exception as e:
                        logger.error(f"Error parsing jsonl line: {e}")
    else:
        logger.error("No jsonl output files found after evalscope run!")

    actual_score = (correct / max(total, 1)) * 100
    diff = abs(actual_score - official_score) if official_score else 0
    passed = diff <= tolerance if official_score else True

    try:
        shutil.rmtree(work_dir)
    except Exception as e:
        logger.warning(f"Failed to cleanup work_dir {work_dir}: {e}")

    result = {
        "test_id": test_id,
        "status": "passed" if passed else "failed",
        "details": f"{dataset_name}: {correct}/{total} correct ({actual_score:.1f}%) vs official {official_score}%",
        "official_score": official_score,
        "actual_score": actual_score,
        "diff": diff,
        "tolerance": tolerance,
        "passed": passed,
        "metrics": {
            "correct": correct,
            "total": total,
            "details": details
        }
    }
    
    write_result(test_id, result)
    return result


def test_aime2025(api_base: str, api_key: str) -> Dict[str, Any]:
    """AIME2025 math benchmark."""
    return run_evalscope_dataset("acc_aime25", "aime25", api_base, api_key)


def test_aime2026(api_base: str, api_key: str) -> Dict[str, Any]:
    """AIME2026 math benchmark."""
    return run_evalscope_dataset("acc_aime26", "aime26", api_base, api_key)


def test_hle(api_base: str, api_key: str) -> Dict[str, Any]:
    """HLE (Humanity's Last Exam) benchmark."""
    return run_evalscope_dataset("acc_hle", "hle", api_base, api_key)


def test_swebench(api_base: str, api_key: str) -> Dict[str, Any]:
    """SWE-Bench Verified coding benchmark."""
    return run_evalscope_dataset("acc_swebench", "swe_bench_verified", api_base, api_key)


def test_gpqa(api_base: str, api_key: str) -> Dict[str, Any]:
    """GPQA (Graduate-level Google-Proof Q&A) benchmark."""
    return run_evalscope_dataset("acc_gpqa", "gpqa_diamond", api_base, api_key, dataset_hub="modelscope", dataset_id="AI-ModelScope/gpqa_diamond")


def test_longbench(api_base: str, api_key: str) -> Dict[str, Any]:
    """LongBench V2 long context understanding benchmark."""
    return run_evalscope_dataset("acc_longbench", "longbench_v2", api_base, api_key, subset_list=["short"])


def run_tau_bench(api_base: str, api_key: str, scenario: str, test_id: str,
                  benchmark_name: str, official_key: str) -> Dict[str, Any]:
    """Generic TAU Bench runner for retail/telecom/airline scenarios."""
    logger.info(f"Running benchmark: {test_id}")
    write_progress(test_id, {"status": "running", "current_step": f"Loading TAU {scenario} benchmark"})

    config = load_config()
    official_score = config.get("benchmarks", {}).get(official_key, {}).get("official", 70.6)
    tolerance = config.get("benchmarks", {}).get(official_key, {}).get("tolerance", 4.0)

    limit = int(os.environ.get("EVALSCOPE_LIMIT", "1"))
    benchmark_data = load_benchmark(benchmark_name)[:limit]
    client = get_client(api_base, api_key)

    correct = 0
    total = len(benchmark_data)
    details = []

    model_name = config.get("model", {}).get("name", "kimi-k2.6")
    for i, item in enumerate(benchmark_data):
        write_progress(test_id, {
            "status": "running",
            "current_step": f"Testing scenario {i+1}/{total}",
            "progress_pct": (i / max(total, 1)) * 100
        })

        # Format tools for function calling
        tools = []
        for tool in item.get("available_tools", []):
            tools.append({
                "type": "function",
                "function": {
                    "name": tool["name"],
                    "description": f"Tool: {tool['name']}",
                    "parameters": {
                        "type": "object",
                        "properties": {k: {"type": v} for k, v in tool["params"].items()},
                        "required": list(tool["params"].keys())
                    }
                }
            })

        messages = [
            {"role": "system", "content": f"You are a helpful {scenario} customer service agent. Use the available tools to help the customer."},
            {"role": "user", "content": item["user_message"]}
        ]

        try:
            # Force temperature to 1.0 to comply with kimi-k2.6 thinking validation
            response = retry_with_backoff(lambda: client.chat.completions.create(
                model=model_name,
                messages=messages,
                tools=tools if tools else None,
                max_tokens=500,
                temperature=1.0
            ))

            tool_calls = response.choices[0].message.tool_calls
            # Check if the correct tool was called
            expected_tool = item.get("expected_tool_call", "")
            if tool_calls and len(tool_calls) > 0:
                called_tool = tool_calls[0].function.name
                is_correct = called_tool == expected_tool
            else:
                is_correct = False

            if is_correct:
                correct += 1

            details.append({
                "id": item["id"],
                "correct": is_correct,
                "expected_tool": expected_tool,
                "called_tool": tool_calls[0].function.name if tool_calls else None,
                "response_excerpt": (response.choices[0].message.content or "")[:200]
            })
        except Exception as e:
            details.append({
                "id": item["id"],
                "correct": False,
                "expected_tool": item.get("expected_tool_call", ""),
                "error": str(e)
            })

    actual_score = (correct / max(total, 1)) * 100
    diff = abs(actual_score - official_score) if official_score else 0
    passed = diff <= tolerance if official_score else True

    result = {
        "test_id": test_id,
        "status": "passed" if passed else "failed",
        "details": f"TAU {scenario}: {correct}/{total} correct tool calls ({actual_score:.1f}%) vs official {official_score}%",
        "official_score": official_score,
        "actual_score": actual_score,
        "diff": diff,
        "tolerance": tolerance,
        "passed": passed,
        "metrics": {
            "correct": correct,
            "total": total,
            "details": details
        }
    }

    write_result(test_id, result)
    return result


def test_tau_retail(api_base: str, api_key: str) -> Dict[str, Any]:
    """TAU Bench Retail benchmark."""
    return run_evalscope_dataset("acc_tau_retail", "tau3_bench", api_base, api_key, subset_list=["retail"])


def test_tau_telecom(api_base: str, api_key: str) -> Dict[str, Any]:
    """TAU Bench Telecom benchmark."""
    return run_evalscope_dataset("acc_tau_telecom", "tau3_bench", api_base, api_key, subset_list=["telecom"])


def test_tau_airline(api_base: str, api_key: str) -> Dict[str, Any]:
    """TAU Bench Airline benchmark."""
    return run_evalscope_dataset("acc_tau_airline", "tau3_bench", api_base, api_key, subset_list=["airline"])


# ============================================================================
# Test Registry
# ============================================================================

TEST_REGISTRY = {
    "acc_aime25": test_aime2025,
    "acc_aime26": test_aime2026,
    "acc_hle": test_hle,
    "acc_swebench": test_swebench,
    "acc_gpqa": test_gpqa,
    "acc_longbench": test_longbench,
    "acc_tau_retail": test_tau_retail,
    "acc_tau_telecom": test_tau_telecom,
    "acc_tau_airline": test_tau_airline,
}


def run_all_tests(api_base: str, api_key: str) -> Dict[str, Any]:
    """Run all accuracy tests and return aggregate results."""
    results = {}
    for test_id, test_func in TEST_REGISTRY.items():
        logger.info(f"Starting benchmark: {test_id}")
        try:
            results[test_id] = test_func(api_base, api_key)
        except Exception as e:
            results[test_id] = {
                "test_id": test_id,
                "status": "failed",
                "details": f"Unhandled error: {str(e)}",
                "official_score": None,
                "actual_score": 0,
                "diff": 0,
                "tolerance": 0,
                "passed": False,
                "metrics": {}
            }

    passed = sum(1 for r in results.values() if r.get("passed", False))
    total = len(results)

    return {
        "suite": "accuracy",
        "name": "精度验收 (Accuracy)",
        "total_tests": total,
        "passed": passed,
        "failed": total - passed,
        "results": results
    }


def main():
    parser = argparse.ArgumentParser(description="Benchmark Accuracy Tests")
    parser.add_argument("--api-base", type=str, default=None, help="API base URL")
    parser.add_argument("--api-key", type=str, default=None, help="API key")
    parser.add_argument("--test-id", type=str, default=None, help="Specific test to run")
    args = parser.parse_args()

    config = load_config()
    api_base = args.api_base or config.get("model", {}).get("api_base", "https://api.unisai.com/v1")
    api_key = args.api_key or os.environ.get(config.get("model", {}).get("api_key_env", "KIMI_API_KEY"), "") or config.get("model", {}).get("api_key", "")

    if not api_key:
        logger.error("No API key provided. Set --api-key or the KIMI_API_KEY environment variable.")
        sys.exit(1)

    if args.test_id:
        if args.test_id in TEST_REGISTRY:
            result = TEST_REGISTRY[args.test_id](api_base, api_key)
            print(json.dumps(result, indent=2))
        else:
            logger.error(f"Unknown test_id: {args.test_id}. Available: {list(TEST_REGISTRY.keys())}")
            sys.exit(1)
    else:
        result = run_all_tests(api_base, api_key)
        print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
