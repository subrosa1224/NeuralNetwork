"""成员 4 测试入口（不依赖 pytest）。

用法（在项目根目录）::

    python run_tests.py

如果安装了 pytest，也可以::

    pytest tests -v
"""

import importlib
import sys
import time
import traceback
from pathlib import Path

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

MODULES = (
    "tests.test_gradient_check",
    "tests.test_losses",
    "tests.test_metrics",
)


def _iter_tests(module):
    tests = [
        (name, obj)
        for name, obj in vars(module).items()
        if name.startswith("test_")
        and callable(obj)
        and getattr(obj, "__module__", None) == module.__name__
    ]
    tests.sort(key=lambda item: item[1].__code__.co_firstlineno)
    return tests


def main():
    total = 0
    failed = 0
    failures = []
    start = time.perf_counter()

    for module_name in MODULES:
        module = importlib.import_module(module_name)
        tests = _iter_tests(module)
        print(f"\n{'=' * 68}")
        print(f"{module_name}  ({len(tests)} 个测试)")
        print("=" * 68)

        for name, fn in tests:
            total += 1
            began = time.perf_counter()
            try:
                fn()
            except Exception:  # noqa: BLE001 - 汇总所有失败
                failed += 1
                failures.append((module_name, name, traceback.format_exc()))
                print(f"[FAIL] {name}  ({time.perf_counter() - began:.3f}s)")
            else:
                print(f"[PASS] {name}  ({time.perf_counter() - began:.3f}s)")

    elapsed = time.perf_counter() - start
    if failures:
        print(f"\n{'=' * 68}")
        print("失败详情")
        print("=" * 68)
        for module_name, name, detail in failures:
            print(f"\n--- {module_name}::{name} ---")
            print(detail.rstrip())

    print(f"\n{'=' * 68}")
    print(
        f"测试完成：{total} 个测试，通过 {total - failed} 个，"
        f"失败 {failed} 个，用时 {elapsed:.2f}s"
    )
    print("=" * 68)
    return 0 if failed == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
