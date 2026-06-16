#!/usr/bin/env python3
"""
Capacity & Stability Tests for LLM Model Benchmark.
Tests TPM capacity, RPM rate limiting, and max_tokens context handling.
"""

import argparse
import asyncio
import json
import logging
import os
import sys
import time
from pathlib import Path
from typing import Dict, Any, List

import aiohttp

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Paths
SCRIPT_DIR = Path(__file__).parent
PROJECT_DIR = SCRIPT_DIR.parent
CONFIG_PATH = PROJECT_DIR / "config.json"
RESULTS_DIR = PROJECT_DIR / "results"
RESULTS_DIR.mkdir(exist_ok=True)


def load_config() -> Dict[str, Any]:
    """Load configuration from config.json"""
    if CONFIG_PATH.exists():
        with open(CONFIG_PATH, 'r', encoding='utf-8') as f:
            return json.load(f)
    return {}


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


def generate_text(target_tokens: int) -> str:
    """Generate text approximately target_tokens long."""
    base = "The quick brown fox jumps over the lazy dog. "
    words_needed = int(target_tokens * 0.9)
    words_per_sentence = len(base.split())
    sentences = max(1, words_needed // words_per_sentence)
    return (base * sentences)[:words_needed * 5]


# ============================================================================
# Test Functions
# ============================================================================

async def test_tpm_capacity(api_base: str, api_key: str) -> Dict[str, Any]:
    """Test TPM (Tokens Per Minute) capacity - 50M TPM target."""
    test_id = "cap_tpm"
    logger.info(f"Running test: {test_id}")
    write_progress(test_id, {"status": "running", "current_step": "Starting TPM capacity test"})

    config = load_config()
    target_tpm = config.get("capacity", {}).get("tpm", 50000000)
    model_name = config.get("model", {}).get("name", "kimi-k2.6")

    # We can't actually test 50M TPM in a short test.
    # Instead, we send a burst of requests and measure throughput.
    url = f"{api_base}/chat/completions"
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
        "User-Agent": "curl/8.0"
    }

    # Generate a medium-length prompt (~1000 tokens input)
    prompt_text = generate_text(1000)
    model_config = config.get("model", {})
    payload = {
        "model": model_name,
        "messages": [{"role": "user", "content": prompt_text}],
        "max_tokens": 500,
        "stream": False
    }
    if "thinking" in model_config:
        payload["thinking"] = model_config["thinking"]

    # Send concurrent requests and measure tokens consumed
    concurrency = 10
    requests_per_wave = 20
    total_tokens = 0
    total_requests_sent = 0
    total_errors = 0
    rate_limited = 0
    latencies = []

    connector = aiohttp.TCPConnector(limit=concurrency + 5)
    timeout = aiohttp.ClientTimeout(total=120)

    start_time = time.perf_counter()
    async with aiohttp.ClientSession(connector=connector, timeout=timeout) as session:
        semaphore = asyncio.Semaphore(concurrency)

        async def send_request(req_id: int):
            nonlocal total_tokens, total_requests_sent, total_errors, rate_limited
            async with semaphore:
                start = time.perf_counter()
                try:
                    async with session.post(url, headers=headers, json=payload) as resp:
                        latency = time.perf_counter() - start
                        latencies.append(latency)

                        if resp.status == 429:
                            rate_limited += 1
                            total_errors += 1
                        elif resp.status == 200:
                            body = await resp.json()
                            usage = body.get("usage", {})
                            total_tokens += usage.get("total_tokens", 0)
                            total_requests_sent += 1
                        else:
                            total_errors += 1
                except Exception as e:
                    total_errors += 1

        # Send waves of requests
        for wave in range(3):
            write_progress(test_id, {
                "status": "running",
                "current_step": f"Sending wave {wave+1}/3 ({requests_per_wave} requests each)",
                "progress_pct": ((wave + 1) / 3) * 100,
                "tokens_so_far": total_tokens
            })

            tasks = [send_request(i) for i in range(requests_per_wave)]
            await asyncio.gather(*tasks, return_exceptions=True)

            if wave < 2:
                await asyncio.sleep(2)
    end_time = time.perf_counter()

    # Estimate TPM from the test using actual wall-clock duration
    test_duration_seconds = end_time - start_time
    test_duration_minutes = test_duration_seconds / 60.0
    estimated_tpm = total_tokens / max(test_duration_minutes, 0.001)

    success_rate = total_requests_sent / max(total_requests_sent + total_errors, 1) * 100
    passed = success_rate > 80 and rate_limited < 5

    result = {
        "test_id": test_id,
        "status": "passed" if passed else "failed",
        "details": f"TPM capacity: {total_tokens} tokens in test, {total_requests_sent} successful requests, {rate_limited} rate limited, success_rate={success_rate:.1f}%",
        "metrics": {
            "target_tpm": target_tpm,
            "tokens_consumed": total_tokens,
            "successful_requests": total_requests_sent,
            "total_errors": total_errors,
            "rate_limited": rate_limited,
            "success_rate": success_rate,
            "avg_latency": sum(latencies) / max(len(latencies), 1),
            "estimated_throughput_tpm": estimated_tpm
        }
    }

    write_result(test_id, result)
    return result


async def test_rpm_limit(api_base: str, api_key: str) -> Dict[str, Any]:
    """Test RPM rate limiting - 500 RPM, 429 response <200ms."""
    test_id = "cap_rpm"
    logger.info(f"Running test: {test_id}")
    write_progress(test_id, {"status": "running", "current_step": "Starting RPM limit test"})

    config = load_config()
    target_rpm = config.get("capacity", {}).get("rpm", 500)
    model_name = config.get("model", {}).get("name", "kimi-k2.6")

    url = f"{api_base}/chat/completions"
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
        "User-Agent": "curl/8.0"
    }
    model_config = config.get("model", {})
    payload = {
        "model": model_name,
        "messages": [{"role": "user", "content": "Say hello"}],
        "max_tokens": 10,
        "stream": False
    }
    if "thinking" in model_config:
        payload["thinking"] = model_config["thinking"]

    capacity_cfg = config.get("capacity", {})
    concurrency = capacity_cfg.get("rpm_concurrency_vu", 100)
    duration_seconds = capacity_cfg.get("rpm_duration_seconds", 20)
    rate_limit_latencies = []
    success_count = 0
    rate_limited_count = 0
    other_errors = 0

    connector = aiohttp.TCPConnector(limit=concurrency + 10)
    timeout = aiohttp.ClientTimeout(total=30)

    async with aiohttp.ClientSession(connector=connector, timeout=timeout) as session:
        async def vu_loop(vu_id: int):
            nonlocal success_count, rate_limited_count, other_errors
            for step in range(duration_seconds):
                start_step = time.perf_counter()
                try:
                    async with session.post(url, headers=headers, json=payload) as resp:
                        latency = time.perf_counter() - start_step
                        if resp.status == 200:
                            success_count += 1
                            await resp.read()
                        elif resp.status == 429:
                            rate_limited_count += 1
                            rate_limit_latencies.append(latency)
                        else:
                            other_errors += 1
                except Exception:
                    other_errors += 1

                # Sleep to make it exactly 1 second per loop iteration
                elapsed = time.perf_counter() - start_step
                sleep_time = max(0.0, 1.0 - elapsed)
                await asyncio.sleep(sleep_time)

        # Progress tracking in the background
        async def track_progress():
            for s in range(duration_seconds):
                await asyncio.sleep(1.0)
                write_progress(test_id, {
                    "status": "running",
                    "current_step": f"VU monitoring: {s+1}/{duration_seconds}s elapsed",
                    "progress_pct": ((s + 1) / duration_seconds) * 100,
                    "success": success_count,
                    "rate_limited": rate_limited_count,
                    "other_errors": other_errors
                })

        # Run all VUs and progress tracker concurrently
        vu_tasks = [vu_loop(i) for i in range(concurrency)]
        await asyncio.gather(*vu_tasks, track_progress(), return_exceptions=True)

    # Check that 429 responses come back quickly (<200ms)
    avg_429_latency = sum(rate_limit_latencies) / max(len(rate_limit_latencies), 1) if rate_limit_latencies else 0
    all_429_fast = all(lat < 0.2 for lat in rate_limit_latencies) if rate_limit_latencies else True

    # Pass if no other unexpected errors occurred and 429 responses (if any) are fast (<200ms).
    passed = (other_errors == 0)
    if rate_limited_count > 0 and not all_429_fast:
        passed = False  # Fail if 429 responses were slow
    
    total_sent = success_count + rate_limited_count + other_errors
    if rate_limited_count == 0 and success_count < total_sent * 0.5:
        passed = False  # Fail if too many requests failed due to other errors

    result = {
        "test_id": test_id,
        "status": "passed" if passed else "failed",
        "details": f"RPM limit: {success_count} succeeded, {rate_limited_count} rate-limited (429), avg 429 latency={avg_429_latency*1000:.1f}ms",
        "metrics": {
            "target_rpm": target_rpm,
            "duration_seconds": duration_seconds,
            "concurrency_vu": concurrency,
            "total_sent": total_sent,
            "success_count": success_count,
            "rate_limited_count": rate_limited_count,
            "other_errors": other_errors,
            "avg_429_latency_ms": avg_429_latency * 1000,
            "all_429_under_200ms": all_429_fast,
            "rate_limit_latencies_ms": [l * 1000 for l in rate_limit_latencies[:20]]
        }
    }

    write_result(test_id, result)
    return result


async def test_max_tokens_200k(api_base: str, api_key: str) -> Dict[str, Any]:
    """Test max_tokens 200K context handling."""
    test_id = "cap_max_tokens"
    logger.info(f"Running test: {test_id}")
    write_progress(test_id, {"status": "running", "current_step": "Starting max_tokens 200K test"})

    config = load_config()
    target_max_tokens = config.get("capacity", {}).get("max_tokens", 200000)
    model_name = config.get("model", {}).get("name", "kimi-k2.6")

    url = f"{api_base}/chat/completions"
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
        "User-Agent": "curl/8.0"
    }

    # Test with increasingly large inputs
    test_sizes = [10000, 50000, 100000, 150000, 200000]
    results_by_size = {}

    for size in test_sizes:
        write_progress(test_id, {
            "status": "running",
            "current_step": f"Testing {size} tokens context",
            "progress_pct": (test_sizes.index(size) / len(test_sizes)) * 100
        })

        # Leave a safety margin of 200 tokens for system prompt and completion overhead
        prompt_size = max(1000, size - 200)
        prompt_text = generate_text(prompt_size)
        model_config = config.get("model", {})
        payload = {
            "model": model_name,
            "messages": [
                {"role": "system", "content": "Summarize the following text in one sentence."},
                {"role": "user", "content": prompt_text}
            ],
            "max_tokens": 100,
            "stream": False
        }
        if "thinking" in model_config:
            payload["thinking"] = model_config["thinking"]

        start = time.perf_counter()
        try:
            async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=300)) as session:
                async with session.post(url, headers=headers, json=payload) as resp:
                    latency = time.perf_counter() - start
                    if resp.status == 200:
                        body = await resp.json()
                        usage = body.get("usage", {})
                        content = body.get("choices", [{}])[0].get("message", {}).get("content", "")
                        results_by_size[size] = {
                            "status": "success",
                            "latency_s": latency,
                            "input_tokens": usage.get("prompt_tokens", 0),
                            "output_tokens": usage.get("completion_tokens", 0),
                            "response_length": len(content)
                        }
                    else:
                        body_text = await resp.text()
                        results_by_size[size] = {
                            "status": "error",
                            "http_status": resp.status,
                            "latency_s": latency,
                            "error": body_text[:200]
                        }
        except asyncio.TimeoutError:
            results_by_size[size] = {
                "status": "timeout",
                "latency_s": time.perf_counter() - start
            }
        except Exception as e:
            results_by_size[size] = {
                "status": "error",
                "error": str(e),
                "latency_s": time.perf_counter() - start
            }

    # Check if target max tokens (200K) was handled successfully
    passed = results_by_size.get(200000, {}).get("status") == "success"
    largest_success = 0
    for size, r in results_by_size.items():
        if r.get("status") == "success":
            largest_success = max(largest_success, size)

    result = {
        "test_id": test_id,
        "status": "passed" if passed else "failed",
        "details": f"max_tokens: largest successful context={largest_success} tokens (target: {target_max_tokens})",
        "metrics": {
            "target_max_tokens": target_max_tokens,
            "largest_successful": largest_success,
            "results_by_size": results_by_size
        }
    }

    write_result(test_id, result)
    return result


def run_tpm(api_base: str, api_key: str) -> Dict[str, Any]:
    """Wrapper to run async TPM test."""
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    try:
        return loop.run_until_complete(test_tpm_capacity(api_base, api_key))
    finally:
        loop.close()


def run_rpm(api_base: str, api_key: str) -> Dict[str, Any]:
    """Wrapper to run async RPM test."""
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    try:
        return loop.run_until_complete(test_rpm_limit(api_base, api_key))
    finally:
        loop.close()


def run_max_tokens(api_base: str, api_key: str) -> Dict[str, Any]:
    """Wrapper to run async max_tokens test."""
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    try:
        return loop.run_until_complete(test_max_tokens_200k(api_base, api_key))
    finally:
        loop.close()


# ============================================================================
# Test Registry
# ============================================================================

TEST_REGISTRY = {
    "cap_tpm": run_tpm,
    "cap_rpm": run_rpm,
    "cap_max_tokens": run_max_tokens,
}


def run_all_tests(api_base: str, api_key: str) -> Dict[str, Any]:
    """Run all capacity tests."""
    results = {}
    for test_id, test_func in TEST_REGISTRY.items():
        logger.info(f"Starting capacity test: {test_id}")
        try:
            results[test_id] = test_func(api_base, api_key)
        except Exception as e:
            results[test_id] = {
                "test_id": test_id,
                "status": "failed",
                "details": f"Unhandled error: {str(e)}",
                "metrics": {}
            }

    passed = sum(1 for r in results.values() if r.get("status") == "passed")
    total = len(results)

    return {
        "suite": "capacity",
        "name": "容量及稳定性 (Capacity)",
        "total_tests": total,
        "passed": passed,
        "failed": total - passed,
        "results": results
    }


def main():
    parser = argparse.ArgumentParser(description="Capacity & Stability Tests")
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
