#!/usr/bin/env python3
"""
Performance/TTFT/OTPS Tests for LLM Model Benchmark.
Measures Time to First Token (TTFT), Output Tokens Per Second (OTPS),
and runs gradient load testing with configurable concurrency.
"""

import argparse
import asyncio
import json
import logging
import os
import statistics
import sys
import time
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple

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

# Prompt templates for different input sizes
PROMPT_SHORT = "What is 2+2? Answer in one word."
PROMPT_MEDIUM = "Write a paragraph about artificial intelligence and its impact on society."
PROMPT_LONG_TEMPLATE = "The following is a technical document.\n\n{text}\n\nSummarize the key points in 3 sentences."

# Generate filler text for long context tests
def generate_filler_text(target_tokens: int) -> str:
    """Generate filler text approximately target_tokens long."""
    # Approx 1.3 tokens per word, so target_tokens * 0.77 words
    words_needed = int(target_tokens * 0.77)
    base_sentence = "The quick brown fox jumps over the lazy dog while contemplating the nature of existence and computational complexity theory. "
    sentence_words = len(base_sentence.split())
    sentences_needed = max(1, words_needed // sentence_words)
    return (base_sentence * sentences_needed)[:words_needed * 5]  # rough char limit


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


async def make_streaming_request(
    session: aiohttp.ClientSession,
    api_base: str,
    api_key: str,
    messages: List[Dict],
    max_tokens: int = 500,
    request_id: int = 0,
    is_ttft_only: bool = False
) -> Dict[str, Any]:
    """Make a single streaming request and measure TTFT and OTPS."""
    config = load_config()
    model_name = config.get("model", {}).get("name", "kimi-k2.6")
    url = f"{api_base}/chat/completions"
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
        "Accept-Encoding": "gzip, deflate"
    }
    model_config = config.get("model", {})
    payload = {
        "model": model_name,
        "messages": messages,
        "max_tokens": max_tokens,
        "stream": True
    }
    if "thinking" in model_config:
        payload["thinking"] = model_config["thinking"]

    result = {
        "request_id": request_id,
        "success": False,
        "ttft": None,
        "total_time": None,
        "output_tokens": 0,
        "otps": None,
        "error": None,
        "status_code": None
    }

    start_time = time.perf_counter()
    first_reasoning_time = None
    first_content_time = None
    token_count = 0

    try:
        async with session.post(url, headers=headers, json=payload, timeout=aiohttp.ClientTimeout(total=120)) as resp:
            result["status_code"] = resp.status

            if resp.status != 200:
                body = await resp.text()
                result["error"] = f"HTTP {resp.status}: {body[:200]}"
                return result

            async for line in resp.content:
                line = line.decode('utf-8', errors='replace').strip()
                if not line.startswith('data: '):
                    continue
                data_str = line[6:]
                if data_str == '[DONE]':
                    break

                try:
                    data = json.loads(data_str)
                    if data.get('choices'):
                        delta = data['choices'][0].get('delta', {})
                        if delta.get('reasoning_content') and first_reasoning_time is None:
                            first_reasoning_time = time.perf_counter()
                        if delta.get('content') and first_content_time is None:
                            first_content_time = time.perf_counter()

                        content = delta.get('content', '') or ''
                        if content:
                            # Approximate token count (rough: 1 token per ~4 chars for English)
                            token_count += max(1, len(content) // 4)

                    # Check for usage in last chunk
                    if data.get('usage'):
                        usage = data['usage']
                        token_count = usage.get('completion_tokens', token_count)
                except json.JSONDecodeError:
                    pass

                # If this is a pure TTFT test, and we have received the first token (reasoning or content), abort immediately!
                if is_ttft_only and (first_reasoning_time is not None or first_content_time is not None):
                    token_count = 1
                    break

        end_time = time.perf_counter()
        result["success"] = True
        result["total_time"] = end_time - start_time
        
        # TTFT: first reasoning token (thinking model starts "thinking" before "speaking")
        first_token = first_reasoning_time or first_content_time
        if first_token is not None:
            result["ttft"] = first_token - start_time
        else:
            result["ttft"] = None
            
        result["output_tokens"] = token_count

        if not is_ttft_only:
            if result["ttft"] is not None and result["ttft"] > 0 and token_count > 0:
                generation_time = result["total_time"] - result["ttft"]
                if generation_time > 0:
                    result["otps"] = token_count / generation_time
            elif token_count > 0 and result["total_time"] > 0:
                result["otps"] = token_count / result["total_time"]

    except asyncio.TimeoutError:
        result["error"] = "Request timed out (120s)"
    except Exception as e:
        result["error"] = str(e)

    return result


async def run_concurrent_test(
    api_base: str,
    api_key: str,
    messages: List[Dict],
    concurrency: int,
    total_requests: int,
    max_tokens: int = 500,
    test_id: str = "perf",
    warmup_requests: int = 0,
    is_ttft_only: bool = False
) -> Dict[str, Any]:
    """Run concurrent streaming requests and collect metrics."""
    connector = aiohttp.TCPConnector(limit=concurrency + 10)
    timeout = aiohttp.ClientTimeout(total=180)

    async with aiohttp.ClientSession(connector=connector, timeout=timeout) as session:
        # Warmup phase
        if warmup_requests > 0:
            logger.info(f"Warming up with {warmup_requests} requests...")
            warmup_tasks = [
                make_streaming_request(session, api_base, api_key, messages, max_tokens, i, is_ttft_only)
                for i in range(warmup_requests)
            ]
            await asyncio.gather(*warmup_tasks, return_exceptions=True)
            logger.info("Warmup complete.")

        # Main test
        results = []
        semaphore = asyncio.Semaphore(concurrency)

        async def bounded_request(req_id: int):
            async with semaphore:
                return await make_streaming_request(session, api_base, api_key, messages, max_tokens, req_id, is_ttft_only)

        tasks = [bounded_request(i) for i in range(total_requests)]

        completed = 0
        for coro in asyncio.as_completed(tasks):
            result = await coro
            results.append(result)
            completed += 1
            if completed % 5 == 0:
                write_progress(test_id, {
                    "status": "running",
                    "current_step": f"Completed {completed}/{total_requests} requests (concurrency={concurrency})",
                    "progress_pct": (completed / total_requests) * 100
                })

    return results


def compute_stats(results: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Compute performance statistics from results."""
    successful = [r for r in results if r["success"]]
    failed = [r for r in results if not r["success"]]

    if not successful:
        return {
            "total_requests": len(results),
            "successful": 0,
            "failed": len(failed),
            "success_rate": 0,
            "ttft": {},
            "otps": {},
            "total_time": {},
            "errors": [r["error"] for r in failed[:5]]
        }

    ttfts = [r["ttft"] for r in successful if r["ttft"] is not None]
    otps_vals = [r["otps"] for r in successful if r["otps"] is not None and r["otps"] > 0]
    total_times = [r["total_time"] for r in successful if r["total_time"] is not None]

    def percentile(data, p):
        if not data:
            return 0
        sorted_data = sorted(data)
        idx = int(len(sorted_data) * p / 100)
        idx = min(idx, len(sorted_data) - 1)
        return sorted_data[idx]

    stats = {
        "total_requests": len(results),
        "successful": len(successful),
        "failed": len(failed),
        "success_rate": len(successful) / len(results) * 100,
        "ttft": {
            "p50": percentile(ttfts, 50) if ttfts else 0,
            "p90": percentile(ttfts, 90) if ttfts else 0,
            "avg": statistics.mean(ttfts) if ttfts else 0,
            "min": min(ttfts) if ttfts else 0,
            "max": max(ttfts) if ttfts else 0
        },
        "otps": {
            "p50": percentile(otps_vals, 50) if otps_vals else 0,
            "p90": percentile(otps_vals, 90) if otps_vals else 0,
            "avg": statistics.mean(otps_vals) if otps_vals else 0,
            "min": min(otps_vals) if otps_vals else 0,
            "max": max(otps_vals) if otps_vals else 0
        },
        "total_time": {
            "p50": percentile(total_times, 50) if total_times else 0,
            "p90": percentile(total_times, 90) if total_times else 0,
            "avg": statistics.mean(total_times) if total_times else 0,
        },
        "errors": [r["error"] for r in failed[:5]]
    }

    return stats


# ============================================================================
# Test Functions
# ============================================================================

async def _test_ttft(api_base: str, api_key: str, test_id: str, context_range: str,
                     target_tokens: int, max_concurrency: int, sla: Dict) -> Dict[str, Any]:
    """Internal TTFT test runner."""
    logger.info(f"Running TTFT test: {test_id} ({context_range})")
    write_progress(test_id, {"status": "running", "current_step": f"Starting TTFT test for {context_range}"})

    # Generate appropriate prompt
    if target_tokens <= 6000:
        messages = [{"role": "user", "content": PROMPT_MEDIUM + " " + generate_filler_text(target_tokens)}]
    else:
        filler = generate_filler_text(target_tokens)
        messages = [{"role": "user", "content": PROMPT_LONG_TEMPLATE.format(text=filler)}]

    total_requests = max_concurrency * 3  # 3 rounds at max concurrency
    results = await run_concurrent_test(
        api_base, api_key, messages,
        concurrency=max_concurrency,
        total_requests=total_requests,
        max_tokens=200,
        test_id=test_id,
        warmup_requests=2,
        is_ttft_only=True
    )

    stats = compute_stats(results)

    # Check against SLA
    ttft_sla = sla
    p50_ok = stats["ttft"]["p50"] <= ttft_sla.get("p50", 999)
    p90_ok = stats["ttft"]["p90"] <= ttft_sla.get("p90", 999)
    avg_ok = stats["ttft"]["avg"] <= ttft_sla.get("avg", 999)
    passed = p50_ok and p90_ok and avg_ok

    result = {
        "test_id": test_id,
        "status": "passed" if passed else "failed",
        "details": f"TTFT {context_range}: P50={stats['ttft']['p50']:.2f}s P90={stats['ttft']['p90']:.2f}s avg={stats['ttft']['avg']:.2f}s (SLA: P50<{ttft_sla.get('p50')}s P90<{ttft_sla.get('p90')}s avg<{ttft_sla.get('avg')}s)",
        "metrics": stats,
        "sla_check": {
            "p50_ok": p50_ok,
            "p90_ok": p90_ok,
            "avg_ok": avg_ok,
            "sla": ttft_sla
        }
    }

    write_result(test_id, result)
    return result


def test_ttft_6k(api_base: str, api_key: str) -> Dict[str, Any]:
    """TTFT test for <6K tokens."""
    config = load_config()
    sla = config.get("performance", {}).get("ttft_sla", {}).get("<6K", {"p90": 5, "p50": 2, "avg": 2, "max_concurrency": 12})
    return asyncio.get_event_loop().run_until_complete(
        _test_ttft(api_base, api_key, "perf_ttft_6k", "<6K", 4000, sla.get("max_concurrency", 12), sla)
    )


def test_ttft_16k(api_base: str, api_key: str) -> Dict[str, Any]:
    """TTFT test for 6-16K tokens."""
    config = load_config()
    sla = config.get("performance", {}).get("ttft_sla", {}).get("6-16K", {"p90": 5, "p50": 2.5, "avg": 4, "max_concurrency": 10})
    return asyncio.get_event_loop().run_until_complete(
        _test_ttft(api_base, api_key, "perf_ttft_16k", "6-16K", 12000, sla.get("max_concurrency", 10), sla)
    )


def test_ttft_32k(api_base: str, api_key: str) -> Dict[str, Any]:
    """TTFT test for 16-32K tokens."""
    config = load_config()
    sla = config.get("performance", {}).get("ttft_sla", {}).get("16-32K", {"p90": 8, "p50": 4, "avg": 6, "max_concurrency": 8})
    return asyncio.get_event_loop().run_until_complete(
        _test_ttft(api_base, api_key, "perf_ttft_32k", "16-32K", 24000, sla.get("max_concurrency", 8), sla)
    )


def test_ttft_64k(api_base: str, api_key: str) -> Dict[str, Any]:
    """TTFT test for 32-64K tokens."""
    config = load_config()
    sla = config.get("performance", {}).get("ttft_sla", {}).get("32-64K", {"p90": 15, "p50": 8, "avg": 8, "max_concurrency": 6})
    return asyncio.get_event_loop().run_until_complete(
        _test_ttft(api_base, api_key, "perf_ttft_64k", "32-64K", 48000, sla.get("max_concurrency", 6), sla)
    )


def test_ttft_128k(api_base: str, api_key: str) -> Dict[str, Any]:
    """TTFT test for 64-128K tokens."""
    config = load_config()
    sla = config.get("performance", {}).get("ttft_sla", {}).get("64-128K", {"p90": 35, "p50": 15, "avg": 15, "max_concurrency": 4})
    return asyncio.get_event_loop().run_until_complete(
        _test_ttft(api_base, api_key, "perf_ttft_128k", "64-128K", 96000, sla.get("max_concurrency", 4), sla)
    )


def test_ttft_256k(api_base: str, api_key: str) -> Dict[str, Any]:
    """TTFT test for 128-256K tokens."""
    config = load_config()
    sla = config.get("performance", {}).get("ttft_sla", {}).get("128-256K", {"p90": 70, "p50": 30, "avg": 30, "max_concurrency": 3})
    return asyncio.get_event_loop().run_until_complete(
        _test_ttft(api_base, api_key, "perf_ttft_256k", "128-256K", 192000, sla.get("max_concurrency", 3), sla)
    )


async def _test_otps(api_base: str, api_key: str, test_id: str, min_otps: float,
                     concurrency: int, total_requests: int) -> Dict[str, Any]:
    """Internal OTPS test runner."""
    logger.info(f"Running OTPS test: {test_id}")
    write_progress(test_id, {"status": "running", "current_step": f"Starting OTPS test (min={min_otps} tok/s)"})

    messages = [{"role": "user", "content": "Write a detailed essay about the history of computing, from Turing machines to modern AI. Include at least 5 paragraphs."}]

    results = await run_concurrent_test(
        api_base, api_key, messages,
        concurrency=concurrency,
        total_requests=total_requests,
        max_tokens=1000,
        test_id=test_id,
        warmup_requests=2
    )

    stats = compute_stats(results)

    avg_otps = stats["otps"]["avg"]
    p50_otps = stats["otps"]["p50"]
    success_rate = stats["success_rate"]

    # OTPS SLA: success rate > 99% AND avg OTPS >= min
    passed = success_rate > 99 and avg_otps >= min_otps

    result = {
        "test_id": test_id,
        "status": "passed" if passed else "failed",
        "details": f"OTPS: avg={avg_otps:.1f} tok/s, P50={p50_otps:.1f} tok/s, success_rate={success_rate:.1f}% (SLA: >={min_otps} tok/s at >99% success)",
        "metrics": stats,
        "sla_check": {
            "avg_otps": avg_otps,
            "min_otps_required": min_otps,
            "success_rate": success_rate,
            "otps_met": avg_otps >= min_otps,
            "success_rate_met": success_rate > 99
        }
    }

    write_result(test_id, result)
    return result


def test_otps_l1(api_base: str, api_key: str) -> Dict[str, Any]:
    """OTPS L1 test (>=30 tokens/s)."""
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    try:
        return loop.run_until_complete(
            _test_otps(api_base, api_key, "perf_otps_l1", 30.0, concurrency=5, total_requests=20)
        )
    finally:
        loop.close()


def test_otps_l2(api_base: str, api_key: str) -> Dict[str, Any]:
    """OTPS L2 test (>=10 tokens/s)."""
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    try:
        return loop.run_until_complete(
            _test_otps(api_base, api_key, "perf_otps_l2", 10.0, concurrency=10, total_requests=30)
        )
    finally:
        loop.close()


async def _gradient_load_test(api_base: str, api_key: str, test_id: str) -> Dict[str, Any]:
    """Gradient load test: warmup 5min, then increasing concurrency."""
    logger.info(f"Running gradient load test: {test_id}")
    write_progress(test_id, {"status": "running", "current_step": "Starting gradient load test"})

    messages = [{"role": "user", "content": PROMPT_MEDIUM}]

    # Phase 1: Warmup (simplified - 10 requests at concurrency 2)
    write_progress(test_id, {"status": "running", "current_step": "Phase 1: Warmup", "phase": 1})
    warmup_results = await run_concurrent_test(
        api_base, api_key, messages, concurrency=2, total_requests=10, max_tokens=200, test_id=test_id
    )
    warmup_stats = compute_stats(warmup_results)

    # Phase 2: Increasing concurrency levels
    concurrency_levels = [2, 5, 10, 20, 30, 50]
    phase_results = {}

    for level in concurrency_levels:
        write_progress(test_id, {
            "status": "running",
            "current_step": f"Phase 2: Concurrency={level}",
            "phase": 2,
            "current_concurrency": level
        })

        level_results = await run_concurrent_test(
            api_base, api_key, messages, concurrency=level,
            total_requests=level * 2, max_tokens=200, test_id=test_id
        )
        phase_results[str(level)] = compute_stats(level_results)

    # Determine overall pass/fail
    all_phases_ok = all(
        phase_results[str(c)]["success_rate"] > 90
        for c in concurrency_levels
    )

    result = {
        "test_id": test_id,
        "status": "passed" if all_phases_ok else "failed",
        "details": f"Gradient load test: warmup success_rate={warmup_stats['success_rate']:.1f}%, levels tested={len(concurrency_levels)}",
        "metrics": {
            "warmup": warmup_stats,
            "phases": phase_results,
            "concurrency_levels": concurrency_levels
        }
    }

    write_result(test_id, result)
    return result


def test_gradient_load(api_base: str, api_key: str) -> Dict[str, Any]:
    """Gradient load test."""
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    try:
        return loop.run_until_complete(
            _gradient_load_test(api_base, api_key, "perf_gradient_load")
        )
    finally:
        loop.close()


# ============================================================================
# Test Registry
# ============================================================================

TEST_REGISTRY = {
    "perf_ttft_6k": test_ttft_6k,
    "perf_ttft_16k": test_ttft_16k,
    "perf_ttft_32k": test_ttft_32k,
    "perf_ttft_64k": test_ttft_64k,
    "perf_ttft_128k": test_ttft_128k,
    "perf_ttft_256k": test_ttft_256k,
    "perf_otps_l1": test_otps_l1,
    "perf_otps_l2": test_otps_l2,
    "perf_gradient_load": test_gradient_load,
}


def run_all_tests(api_base: str, api_key: str) -> Dict[str, Any]:
    """Run all performance tests."""
    results = {}
    for test_id, test_func in TEST_REGISTRY.items():
        logger.info(f"Starting performance test: {test_id}")
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
        "suite": "perf",
        "name": "性能验收 (Performance)",
        "total_tests": total,
        "passed": passed,
        "failed": total - passed,
        "results": results
    }


def main():
    parser = argparse.ArgumentParser(description="Performance/TTFT/OTPS Tests")
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
