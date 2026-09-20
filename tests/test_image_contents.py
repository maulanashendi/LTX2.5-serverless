"""The image is built from an explicit COPY list, so a new local module is easy to
forget. Forgetting one is invisible until a worker boots and dies on ImportError,
so lock the Dockerfile's file list against what the handler actually imports."""
import ast
import re
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DOCKERFILE = ROOT / "Dockerfile"

# Modules that only the test suite or tooling needs, never the running worker.
NOT_SHIPPED: set[str] = set()


def local_modules() -> set[str]:
    return {p.stem for p in ROOT.glob("*.py")}


def copied_files() -> set[str]:
    """Every path listed on the Dockerfile's root-level COPY lines."""
    copied: set[str] = set()
    for line in DOCKERFILE.read_text().splitlines():
        stripped = line.strip()
        if not stripped.startswith("COPY "):
            continue
        for token in stripped[len("COPY ") :].split():
            if token in ("./", ".", "\\") or token.startswith("--"):
                continue
            copied.add(Path(token).name)
    return copied


def imported_local_modules(source: Path) -> set[str]:
    tree = ast.parse(source.read_text())
    names: set[str] = set()
    known = local_modules()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            names.update(a.name.split(".")[0] for a in node.names)
        elif isinstance(node, ast.ImportFrom) and node.level == 0 and node.module:
            names.add(node.module.split(".")[0])
    return {n for n in names if n in known}


class TestImageContents(unittest.TestCase):
    def test_handler_imports_are_copied_into_the_image(self):
        copied = copied_files()
        # Walk the local import graph starting from the handler.
        seen: set[str] = set()
        queue = ["handler"]
        while queue:
            mod = queue.pop()
            if mod in seen:
                continue
            seen.add(mod)
            queue.extend(imported_local_modules(ROOT / f"{mod}.py"))

        for mod in sorted(seen - NOT_SHIPPED):
            with self.subTest(module=mod):
                self.assertIn(
                    f"{mod}.py",
                    copied,
                    f"{mod}.py is imported by the handler but is not in the "
                    f"Dockerfile COPY list, so the worker will die on ImportError.",
                )

    def test_workflow_templates_are_copied_into_the_image(self):
        copied = copied_files()
        for template in ("video_ltx2_5_i2v_API.json", "video_ltx2_5_t2v_API.json"):
            self.assertIn(template, copied)

    def test_entrypoint_is_copied_and_executable(self):
        text = DOCKERFILE.read_text()
        self.assertIn("start.sh", copied_files())
        self.assertRegex(text, r"CMD \[\"/start\.sh\"\]")


if __name__ == "__main__":
    unittest.main()
