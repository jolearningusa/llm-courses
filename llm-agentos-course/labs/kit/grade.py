#!/usr/bin/env python3
"""Autograder: run one lab's tests against the scripted model and print a score.

    python grade.py 0        # Lab 0 (tests/test_lab00.py); same as `make grade LAB=0`
    python grade.py kit      # the kit's own tests (instructors and CI)

The score is the share of tests that pass. Tests use only the scripted model, so grading is free,
offline and gives the same result on every machine. Exit status is 0 only when every test passes.
"""

from __future__ import annotations

import os
import sys
import unittest
from pathlib import Path

KIT = Path(__file__).resolve().parent


class _Result(unittest.TextTestResult):
    def __init__(self, *a, **kw):
        super().__init__(*a, **kw)
        self.outcomes: list[tuple[str, str]] = []

    def addSuccess(self, test):
        super().addSuccess(test)
        self.outcomes.append(("ok", test.id()))

    def addFailure(self, test, err):
        super().addFailure(test, err)
        self.outcomes.append(("FAIL", test.id()))

    def addError(self, test, err):
        super().addError(test, err)
        kind = "TODO" if err[0] is NotImplementedError else "ERROR"
        self.outcomes.append((kind, test.id()))

    def addSubTest(self, test, subtest, err):
        super().addSubTest(test, subtest, err)
        if err is not None and ("FAIL", test.id()) not in self.outcomes:
            self.outcomes.append(("FAIL", test.id()))


def main(argv: list[str]) -> int:
    if len(argv) != 1:
        print(__doc__)
        return 2
    lab = argv[0]
    name = "test_kit" if lab == "kit" else f"test_lab{int(lab):02d}"
    if not (KIT / "tests" / f"{name}.py").is_file():
        print(f"no tests for lab {lab} yet (tests/{name}.py)")
        return 2
    sys.path.insert(0, str(KIT))
    suite = unittest.defaultTestLoader.loadTestsFromName(f"tests.{name}")
    with open(os.devnull, "w") as quiet:
        result = unittest.TextTestRunner(resultclass=_Result, verbosity=0, stream=quiet).run(suite)
    for kind, test_id in result.outcomes:
        print(f"  {kind:5s} {test_id.split('.', 2)[-1]}")
    for test, tb in result.failures + result.errors:
        if "NotImplementedError" in tb:
            continue
        print(f"\n--- {test.id()}\n{tb.strip()}")
    passed = sum(k == "ok" for k, _ in result.outcomes)
    total = len(result.outcomes)
    print(f"\n{'Lab ' + lab if lab != 'kit' else 'Kit'}: {passed}/{total} tests pass, score {100 * passed // max(total, 1)}")
    return 0 if passed == total else 1


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
