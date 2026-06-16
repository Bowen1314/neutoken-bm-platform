#!/usr/bin/env python3
"""
Unified Test Runner for LLM Model Benchmark.
Accepts a test_id as command line argument, imports and runs the corresponding
test function, and writes results to the results/ directory.
"""

import argparse
import importlib
import json
import logging
import os
import sys
import time
import traceback
import io
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, Optional

# Fix Windows GBK encoding (protect against double-reassignment)
os.environ.setdefault('PYTHONUTF8', '1')
try:
    if sys.stdout.encoding != 'utf-8':
        sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
except (ValueError, AttributeError):
    pass

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Paths
SCRIPT_DIR = Path(__file__).parent
PROJECT_DIR = SCRIPT_DIR.parent
CONFIG_PATH = PROJECT_DIR / "config.json"
RESULTS_DIR = PROJECT_DIR / "results"
TEST_SPEC_PATH = PROJECT_DIR / "test_spec.json"
RESULTS_DIR.mkdir(exist_ok=True)

# Add scripts directory to path for imports
sys.path.insert(0, str(SCRIPT_DIR))


def load_config() -> Dict[str, Any]:
    """Load configuration from config.json"""
    if CONFIG_PATH.exists():
        with open(CONFIG_PATH, 'r', encoding='utf-8') as f:
            return json.load(f)
    return {}


def load_test_spec() -> Dict[str, Any]:
    """Load test specification from test_spec.json"""
    if TEST_SPEC_PATH.exists():
        with open(TEST_SPEC_PATH, 'r') as f:
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


# ============================================================================
# Test ID to Module/Function Mapping
# ============================================================================

# Maps test_id prefix to module name
MODULE_MAP = {
    "eng": "test_engine",
    "acc": "test_accuracy",
    "perf": "test_performance",
    "hall": "test_hallucination",
    "cap": "test_capacity",
    "avail": "test_availability",
}


def get_module_for_test(test_id: str) -> Optional[str]:
    """Get the module name for a given test_id."""
    for prefix, module in MODULE_MAP.items():
        if test_id.startswith(prefix):
            return module
    return None


def run_test(test_id: str, api_base: str, api_key: str) -> Dict[str, Any]:
    """Run a single test by test_id."""
    logger.info(f"Running test: {test_id}")

    write_progress(test_id, {
        "status": "running",
        "current_step": "Initializing test",
        "start_time": datetime.utcnow().isoformat()
    })

    module_name = get_module_for_test(test_id)
    if not module_name:
        error_result = {
            "test_id": test_id,
            "status": "failed",
            "details": f"Unknown test_id prefix: {test_id}",
            "metrics": {},
            "error": "test_id not found in module map"
        }
        write_result(test_id, error_result)
        return error_result

    try:
        # Import the module
        module = importlib.import_module(module_name)

        # Check if test_id is in the module's TEST_REGISTRY
        if hasattr(module, 'TEST_REGISTRY') and test_id in module.TEST_REGISTRY:
            test_func = module.TEST_REGISTRY[test_id]
            result = test_func(api_base, api_key)
        else:
            # Try to find a matching function
            func_name = test_id
            if hasattr(module, func_name):
                test_func = getattr(module, func_name)
                result = test_func(api_base, api_key)
            else:
                error_result = {
                    "test_id": test_id,
                    "status": "failed",
                    "details": f"Test function '{test_id}' not found in module '{module_name}'",
                    "metrics": {},
                    "error": f"Available tests: {list(module.TEST_REGISTRY.keys()) if hasattr(module, 'TEST_REGISTRY') else 'N/A'}"
                }
                write_result(test_id, error_result)
                return error_result

        # Ensure result has test_id
        result["test_id"] = test_id
        result["end_time"] = datetime.utcnow().isoformat()

        write_progress(test_id, {
            "status": "completed",
            "result_status": result.get("status", "unknown"),
            "end_time": result["end_time"]
        })

        write_result(test_id, result)
        return result

    except Exception as e:
        logger.error(f"Test {test_id} failed with error: {e}")
        logger.error(traceback.format_exc())

        error_result = {
            "test_id": test_id,
            "status": "failed",
            "details": f"Unhandled error: {str(e)}",
            "metrics": {},
            "error": str(e),
            "traceback": traceback.format_exc(),
            "end_time": datetime.utcnow().isoformat()
        }

        write_progress(test_id, {
            "status": "error",
            "error": str(e),
            "end_time": error_result["end_time"]
        })

        write_result(test_id, error_result)
        return error_result


def run_suite(suite_id: str, api_base: str, api_key: str) -> Dict[str, Any]:
    """Run all tests in a test suite."""
    logger.info(f"Running suite: {suite_id}")

    test_spec = load_test_spec()
    suites = test_spec.get("test_suites", [])

    suite_config = None
    for suite in suites:
        if suite["id"] == suite_id:
            suite_config = suite
            break

    if not suite_config:
        return {
            "suite": suite_id,
            "status": "failed",
            "details": f"Suite '{suite_id}' not found in test_spec.json",
            "results": {}
        }

    results = {}
    tests = suite_config.get("tests", [])

    for i, test_config in enumerate(tests):
        test_id = test_config["id"]
        write_progress(f"{suite_id}_suite", {
            "status": "running",
            "current_step": f"Running test {i+1}/{len(tests)}: {test_config['name']}",
            "progress_pct": ((i) / len(tests)) * 100,
            "current_test": test_id
        })

        result = run_test(test_id, api_base, api_key)
        results[test_id] = result

    passed = sum(1 for r in results.values() if r.get("status") == "passed")
    total = len(results)

    suite_result = {
        "suite": suite_id,
        "name": suite_config.get("name", suite_id),
        "status": "passed" if passed == total else "failed",
        "total_tests": total,
        "passed": passed,
        "failed": total - passed,
        "results": results,
        "end_time": datetime.utcnow().isoformat()
    }

    write_result(f"{suite_id}_suite", suite_result)
    return suite_result


def run_all(api_base: str, api_key: str) -> Dict[str, Any]:
    """Run all test suites."""
    test_spec = load_test_spec()
    suites = test_spec.get("test_suites", [])

    all_results = {}
    total_passed = 0
    total_tests = 0

    for suite in suites:
        suite_id = suite["id"]
        suite_result = run_suite(suite_id, api_base, api_key)
        all_results[suite_id] = suite_result
        total_passed += suite_result.get("passed", 0)
        total_tests += suite_result.get("total_tests", 0)

    final_result = {
        "status": "completed",
        "total_suites": len(suites),
        "total_tests": total_tests,
        "total_passed": total_passed,
        "total_failed": total_tests - total_passed,
        "suites": all_results,
        "end_time": datetime.utcnow().isoformat()
    }

    write_result("all_tests", final_result)
    return final_result


def list_tests():
    """List all available tests."""
    test_spec = load_test_spec()
    suites = test_spec.get("test_suites", [])

    print("=" * 80)
    print("Available Tests")
    print("=" * 80)

    for suite in suites:
        print(f"\n{'─' * 60}")
        print(f"Suite: {suite['name']} (id: {suite['id']})")
        print(f"{'─' * 60}")
        for test in suite.get("tests", []):
            print(f"  {test['id']:30s} {test['name']}")
            print(f"  {'':30s} {test.get('desc', '')}")

    print(f"\n{'=' * 80}")
    print(f"\nTo run a specific test: python runner.py --test-id <test_id>")
    print(f"To run a suite:         python runner.py --suite <suite_id>")
    print(f"To run all tests:       python runner.py --all")
    print(f"To list tests:          python runner.py --list")


def main():
    parser = argparse.ArgumentParser(description="Unified Test Runner for LLM Benchmark")
    parser.add_argument("--api-base", type=str, default=None, help="API base URL")
    parser.add_argument("--api-key", type=str, default=None, help="API key")
    parser.add_argument("--test-id", type=str, default=None, help="Specific test ID to run")
    parser.add_argument("--suite", type=str, default=None, help="Test suite ID to run")
    parser.add_argument("--all", action="store_true", help="Run all test suites")
    parser.add_argument("--list", action="store_true", help="List all available tests")
    args = parser.parse_args()

    if args.list:
        list_tests()
        return

    config = load_config()
    api_base = args.api_base or config.get("model", {}).get("api_base", "https://api.unisai.com/v1")
    api_key = args.api_key or config.get("model", {}).get("api_key", "") or os.environ.get(config.get("model", {}).get("api_key_env", "KIMI_API_KEY"), "")

    if not api_key:
        logger.error("No API key provided. Set --api-key or the KIMI_API_KEY environment variable.")
        sys.exit(1)

    start_time = time.time()

    if args.test_id:
        result = run_test(args.test_id, api_base, api_key)
        elapsed = time.time() - start_time
        print(f"\n{'=' * 60}")
        print(f"Test: {args.test_id}")
        print(f"Status: {result.get('status', 'unknown')}")
        print(f"Details: {result.get('details', 'N/A')}")
        print(f"Elapsed: {elapsed:.2f}s")
        print(f"{'=' * 60}")

    elif args.suite:
        result = run_suite(args.suite, api_base, api_key)
        elapsed = time.time() - start_time
        print(f"\n{'=' * 60}")
        print(f"Suite: {args.suite}")
        print(f"Status: {result.get('status', 'unknown')}")
        print(f"Passed: {result.get('passed', 0)}/{result.get('total_tests', 0)}")
        print(f"Elapsed: {elapsed:.2f}s")
        print(f"{'=' * 60}")

    elif args.all:
        result = run_all(api_base, api_key)
        elapsed = time.time() - start_time
        print(f"\n{'=' * 60}")
        print(f"All Tests Complete")
        print(f"Total: {result.get('total_passed', 0)}/{result.get('total_tests', 0)} passed")
        print(f"Elapsed: {elapsed:.2f}s")
        print(f"{'=' * 60}")

    else:
        parser.print_help()
        print("\nUse --list to see available tests.")


if __name__ == "__main__":
    main()
