# AGENTS.md

## Scope

- This file applies to the entire repository.
- Use this as the default test-running policy for coding agents.

## Objective

- Run and verify tests in a way that matches the book workflow (`book/src/*.md`).
- Prefer `uv` entrypoints defined in `pyproject.toml`.

## Technology Stack

- **Package manager:** `uv` (not `pdm`)
- **Tensor framework:** PyTorch + CUDA (not MLX)
- **GPU kernels:** PyTorch + Triton (not C++/nanobind extensions)
- When implementing assignments, replace any MLX code with equivalent PyTorch code.
- When writing custom GPU kernels, use Triton instead of C++ extensions.

## Environment Requirements

- Linux with NVIDIA GPU + CUDA is expected.
- Install dependencies first:

```bash
uv sync
uv run python scripts/check-installation.py
```

- Optional baseline check from the setup chapter (reference solution, Week 1):

```bash
uv run python scripts/dev-tools.py test-refsol -- -k week_1
```

## Agent Test Workflow

1. Start with the smallest relevant scope (`--week` + `--day`).
2. Use pytest filters via `-- -k ...` to isolate failing tasks.
3. Run broader suites only after targeted tests pass.
4. If extension code changed, rebuild extensions before testing.

## Canonical Commands

Run all tests:

```bash
uv run python scripts/dev-tools.py test
```

Run a specific chapter/day:

```bash
uv run python scripts/dev-tools.py test --week <WEEK> --day <DAY>
```

Run with pytest filters:

```bash
uv run python scripts/dev-tools.py test --week 1 --day 3 -- -k task_2
uv run python scripts/dev-tools.py test --week 2 --day 2 -- -k cpu
uv run python scripts/dev-tools.py test --week 2 --day 2 -- -k gpu
```

Run reference-solution tests:

```bash
uv run python scripts/dev-tools.py test-refsol
uv run python scripts/dev-tools.py test-refsol --week 2 --day 2 -- -k cpu
```

Run main entrypoints:

```bash
uv run python main.py
uv run python main.py --loader week1
uv run python main.py --loader week2
```

Run benchmarks:

```bash
uv run python bench.py
uv run pytest benches
```

Copy test files:

```bash
uv run python scripts/dev-tools.py copy-test --week <WEEK> --day <DAY> --force
```

Format:

```bash
uv run ruff format
```

## Extension Rebuild Rule

Rebuild before tests if these changed:

- `src/extensions/src/*`

Commands:

```bash
uv run python src/extensions/build.py
```

Note: As we migrate kernels to Triton, C++ extensions will be replaced and this step may no longer be needed.

## Guardrails

- Use `--` before pytest args (`-k`, `-q`, `--collect-only`, etc.).
- `uv run python scripts/dev-tools.py test --week X --day Y` auto-copies `tests_refsol/test_week_X_day_Y.py` into `tests/`.
- Model-dependent tests (0.5B/1.5B/7B) skip when models are not downloaded locally.
- When encountering MLX imports (`import mlx`, `import mlx.core`, `import mlx.nn`), replace with PyTorch equivalents.
- When encountering C++ extension builds (`nanobind`, `src/extensions`), consider replacing with Triton kernels where appropriate.
