# Repository Guidelines

## Project Structure & Module Organization

This repository contains CS149 course materials and assignment work. `asst1/` holds CPU threading, SIMD intrinsics, ISPC, SAXPY, and k-means programs, each in its own `prog*/` directory with a local `Makefile`. `asst2/asst2-master/` contains task-system code in `part_a/` and `part_b/`, shared headers in `common/`, and the test harness in `tests/`. `asst3/` contains CUDA projects in `saxpy/`, `scan/`, and `render/`, plus handout assets. `cs149gpt/` contains the NanoGPT attention assignment, Python drivers, C++/ISPC extension code, model data, and visual assets. `slide/` and `slide_notes/` are reference materials.

## Build, Test, and Development Commands

Build from the specific assignment subdirectory, not the repository root.

- `cd asst1/prog1_mandelbrot_threads && make`: builds `mandelbrot`.
- `cd asst2/asst2-master/part_a && make && ./runtasks`: builds and runs the task-system harness.
- `cd asst3/scan && make`: builds `cudaScan` with `nvcc`.
- `cd asst3/scan && make check_scan`: runs the scan checker.
- `cd asst3/scan && make check_find_repeats`: runs the find-repeats checker.
- `cd cs149gpt && python3 gpt149.py 4Daccess`: tests tensor accessor code.
- `cd cs149gpt && python3 gpt149.py part1`: tests the first attention implementation.

Use `make clean` in subprojects before rebuilding from scratch.

## Coding Style & Naming Conventions

Most native code is C++11, CUDA C++, or ISPC. Follow the surrounding file style: 4-space indentation in C++/CUDA bodies, descriptive camelCase function names such as `mandelbrotThread` and `fourDimRead`, and lowercase executable names. Keep performance-sensitive loops simple and explicit. Prefer existing helpers such as `CycleTimer`, `ppm.cpp`, and assignment-provided intrinsics over new dependencies.

## Testing Guidelines

Run the checker or executable associated with the directory you changed. For CUDA work, use the provided `checker.py` scripts in `asst3/scan/` and `asst3/render/`. For `cs149gpt`, use the named `gpt149.py` parts described in its README. Include correctness output and relevant timing/performance notes when changing optimized code.

## Commit & Pull Request Guidelines

Git history is not readable in this checkout, so use clear, imperative commit subjects such as `Fix scan downsweep indexing` or `Optimize naive attention loop order`. Keep commits scoped to one assignment or feature. Pull requests should describe the changed files, list commands run, note hardware used for performance measurements, and include screenshots only when visual output changes.

## Environment Notes

Some assignments assume specific hardware: `asst1` targets Stanford myth machines with ISPC, `asst3` requires CUDA-capable AWS instances, and `cs149gpt` uses Python/PyTorch extension compilation. Do not commit generated binaries, `.ppm` outputs, logs, or local cache files.
