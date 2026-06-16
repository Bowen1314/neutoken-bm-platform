#!/usr/bin/env python3
"""
Availability Tests for LLM Model Benchmark.
Tests SLA monitoring, RTO (Recovery Time Objective) measurement,
and service availability verification.
"""

import argparse
import asyncio
import json
import logging
import os
import sys
import time
from datetime import datetime, timezone
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


async def health_check(session: aiohttp.ClientSession, api_base: str, api_key: str) -> Dict[str, Any]:
    """Perform a single health check request."""
    config = load_config()
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
        "messages": [{"role": "user", "content": "Hi"}],
        "max_tokens": 5,
        "stream": False
    }
    if "thinking" in model_config:
        payload["thinking"] = model_config["thinking"]

    start = time.perf_counter()
    try:
        async with session.post(url, headers=headers, json=payload,
                               timeout=aiohttp.ClientTimeout(total=30)) as resp:
            latency = time.perf_counter() - start
            body = await resp.read()
            return {
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "status_code": resp.status,
                "available": resp.status == 200,
                "latency_s": latency,
                "error": None
            }
    except asyncio.TimeoutError:
        return {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "status_code": None,
            "available": False,
            "latency_s": time.perf_counter() - start,
            "error": "timeout"
        }
    except Exception as e:
        return {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "status_code": None,
            "available": False,
            "latency_s": time.perf_counter() - start,
            "error": str(e)
        }


# ============================================================================
# Test Functions
# ============================================================================

async def _test_sla_availability(api_base: str, api_key: str, test_id: str) -> Dict[str, Any]:
    """Test service availability SLA (99.9% monthly)."""
    logger.info(f"Running SLA availability test: {test_id}")
    write_progress(test_id, {"status": "running", "current_step": "Starting SLA monitoring"})

    config = load_config()
    sla_monthly = config.get("availability", {}).get("sla_monthly", 99.9)
    unavailable_threshold = config.get("availability", {}).get("unavailable_threshold", 0.1)
    unavailable_window_min = config.get("availability", {}).get("unavailable_window_min", 10)

    # Run health checks for a period (simplified: 60 checks at 10s intervals = 10 min)
    check_interval = int(os.environ.get("AVAIL_CHECK_INTERVAL", "10"))
    total_checks = int(os.environ.get("AVAIL_TOTAL_CHECKS", "60"))
    checks = []

    connector = aiohttp.TCPConnector(limit=5)
    async with aiohttp.ClientSession(connector=connector) as session:
        for i in range(total_checks):
            check = await health_check(session, api_base, api_key)
            checks.append(check)

            available_count = sum(1 for c in checks if c["available"])
            current_availability = available_count / len(checks) * 100

            write_progress(test_id, {
                "status": "running",
                "current_step": f"Health check {i+1}/{total_checks}",
                "progress_pct": ((i + 1) / total_checks) * 100,
                "current_availability": current_availability,
                "last_check": check
            })

            if i < total_checks - 1:
                await asyncio.sleep(check_interval)

    # Calculate availability metrics
    total_time = len(checks) * check_interval
    available_checks = sum(1 for c in checks if c["available"])
    unavailable_checks = len(checks) - available_checks
    availability_pct = available_checks / len(checks) * 100

    latencies = [c["latency_s"] for c in checks if c["available"]]
    avg_latency = sum(latencies) / max(len(latencies), 1) if latencies else 0

    # Find consecutive failures (unavailability windows)
    max_consecutive_failures = 0
    current_failures = 0
    unavailability_windows = []
    window_start = None

    for check in checks:
        if not check["available"]:
            current_failures += 1
            if window_start is None:
                window_start = check["timestamp"]
        else:
            if current_failures > 0:
                unavailability_windows.append({
                    "start": window_start,
                    "end": check["timestamp"],
                    "duration_checks": current_failures,
                    "duration_seconds": current_failures * check_interval
                })
            max_consecutive_failures = max(max_consecutive_failures, current_failures)
            current_failures = 0
            window_start = None

    # Handle trailing failures
    if current_failures > 0:
        unavailability_windows.append({
            "start": window_start,
            "end": checks[-1]["timestamp"],
            "duration_checks": current_failures,
            "duration_seconds": current_failures * check_interval
        })

    # Check SLA
    passed = availability_pct >= sla_monthly

    result = {
        "test_id": test_id,
        "status": "passed" if passed else "failed",
        "details": f"SLA availability: {availability_pct:.2f}% (target: {sla_monthly}%), {available_checks}/{len(checks)} checks passed, avg_latency={avg_latency:.2f}s",
        "metrics": {
            "sla_target": sla_monthly,
            "actual_availability_pct": availability_pct,
            "total_checks": len(checks),
            "available_checks": available_checks,
            "unavailable_checks": unavailable_checks,
            "avg_latency_s": avg_latency,
            "max_consecutive_failures": max_consecutive_failures,
            "unavailability_windows": unavailability_windows,
            "check_interval_s": check_interval,
            "test_duration_s": total_time,
            "sla_met": passed
        }
    }

    write_result(test_id, result)
    return result


async def _test_rto(api_base: str, api_key: str, test_id: str, period: str,
                    target_rto_minutes: int) -> Dict[str, Any]:
    """Test RTO (Recovery Time Objective) for peak/off-peak periods."""
    logger.info(f"Running RTO test: {test_id} ({period})")
    write_progress(test_id, {"status": "running", "current_step": f"Starting RTO test for {period} period"})

    # RTO test: monitor for failures and measure recovery time
    # In a real scenario, this would run continuously. Here we do a snapshot test.
    check_interval = int(os.environ.get("RTO_CHECK_INTERVAL", "5"))
    total_checks = int(os.environ.get("RTO_TOTAL_CHECKS", "120"))
    checks = []

    connector = aiohttp.TCPConnector(limit=5)
    async with aiohttp.ClientSession(connector=connector) as session:
        for i in range(total_checks):
            check = await health_check(session, api_base, api_key)
            checks.append(check)

            write_progress(test_id, {
                "status": "running",
                "current_step": f"RTO monitoring {period}: check {i+1}/{total_checks}",
                "progress_pct": ((i + 1) / total_checks) * 100,
                "available": check["available"]
            })

            if i < total_checks - 1:
                await asyncio.sleep(check_interval)

    # Analyze recovery times
    recovery_times = []
    failure_start = None

    for i, check in enumerate(checks):
        if not check["available"] and failure_start is None:
            failure_start = i
        elif check["available"] and failure_start is not None:
            recovery_time_checks = i - failure_start
            recovery_time_seconds = recovery_time_checks * check_interval
            recovery_times.append({
                "failure_start_check": failure_start,
                "recovery_check": i,
                "recovery_time_s": recovery_time_seconds,
                "recovery_time_min": recovery_time_seconds / 60
            })
            failure_start = None

    # Calculate metrics
    available_count = sum(1 for c in checks if c["available"])
    availability_pct = available_count / len(checks) * 100

    max_recovery_time = max((r["recovery_time_min"] for r in recovery_times), default=0)
    avg_recovery_time = (sum(r["recovery_time_min"] for r in recovery_times) /
                        max(len(recovery_times), 1)) if recovery_times else 0

    # If no failures observed, RTO is theoretically met
    if not recovery_times:
        rto_met = True
        max_recovery_time = 0
    else:
        rto_met = max_recovery_time <= target_rto_minutes

    passed = rto_met and availability_pct >= 99.0

    result = {
        "test_id": test_id,
        "status": "passed" if passed else "failed",
        "details": f"RTO {period}: max_recovery={max_recovery_time:.1f}min (target: {target_rto_minutes}min), availability={availability_pct:.1f}%, failures_detected={len(recovery_times)}",
        "metrics": {
            "period": period,
            "target_rto_minutes": target_rto_minutes,
            "max_recovery_time_min": max_recovery_time,
            "avg_recovery_time_min": avg_recovery_time,
            "recovery_events": len(recovery_times),
            "recovery_times": recovery_times,
            "total_checks": len(checks),
            "availability_pct": availability_pct,
            "rto_met": rto_met,
            "check_interval_s": check_interval
        }
    }

    write_result(test_id, result)
    return result


def test_sla(api_base: str, api_key: str) -> Dict[str, Any]:
    """SLA availability test."""
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    try:
        return loop.run_until_complete(_test_sla_availability(api_base, api_key, "avail_sla"))
    finally:
        loop.close()


def test_rto_peak(api_base: str, api_key: str) -> Dict[str, Any]:
    """RTO peak hours test (8:00-24:00, target <=10min)."""
    config = load_config()
    target_rto = config.get("availability", {}).get("rto_peak", 10)
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    try:
        return loop.run_until_complete(
            _test_rto(api_base, api_key, "avail_rto_peak", "peak", target_rto)
        )
    finally:
        loop.close()


def test_rto_offpeak(api_base: str, api_key: str) -> Dict[str, Any]:
    """RTO off-peak hours test (0:00-8:00, target <=60min)."""
    config = load_config()
    target_rto = config.get("availability", {}).get("rto_offpeak", 60)
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    try:
        return loop.run_until_complete(
            _test_rto(api_base, api_key, "avail_rto_offpeak", "offpeak", target_rto)
        )
    finally:
        loop.close()


# ============================================================================
# Test Registry
# ============================================================================

TEST_REGISTRY = {
    "avail_sla": test_sla,
    "avail_rto_peak": test_rto_peak,
    "avail_rto_offpeak": test_rto_offpeak,
}


def run_all_tests(api_base: str, api_key: str) -> Dict[str, Any]:
    """Run all availability tests."""
    results = {}
    for test_id, test_func in TEST_REGISTRY.items():
        logger.info(f"Starting availability test: {test_id}")
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
        "suite": "availability",
        "name": "可用性验收 (Availability)",
        "total_tests": total,
        "passed": passed,
        "failed": total - passed,
        "results": results
    }


def main():
    parser = argparse.ArgumentParser(description="Availability Tests (SLA/RTO)")
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
