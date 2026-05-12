# CLAUDE.md — RVSPOC P2601 (KleidiCV RISC-V + RVV)

> This file contains project-specific instructions, overriding the global `~/.claude/CLAUDE.md`.

## Project Background

KleidiCV (formerly TensorFlow Lite) to RISC-V, and vectorize all existing ARM Neon/SVE optimized kernels with RVV 1.0.


## Repository Structure

Complete KleidiCV source code. Our work is applied as patches on top of it, without modifying the existing KleidiCV file structure (unless necessary to modify `#ifdef` branches or BUILD configurations).

New RVV kernels are uniformly placed adjacent to the original files  (to be finalized during implementation).

## Development Constraints (from the challenge, mandatory)

- C++17 or above
- RVV implementations should prioritize intrinsics (`<riscv_vector.h>`), supplemented by inline assembly if necessary
- **Must** retain scalar fallback (isolated by conditional compilation `#ifdef __riscv_vector`)
- Must support VLEN 128/256/512 adaptively (using `vsetvl`)
- Build: ssh 192.168.5.211 (username/password openkylin), cd /home/openkylin/github/rvspoc-P2601-kleidicv for local compilation
- License: Apache 2.0 (consistent with KleidiCV)
- **Forbidden** to copy existing third-party RISC-V TFLite ports; implement independently or clearly mark citations
- Using AI to assist in writing code requires explaining the usage and proportion in the final report



## Submission Specifications

- Commit messages must detail changes + why + impact scope (refer to rule 4 in the global CLAUDE.md)
- Record decision-making info in `docs/decisions.md`, debugging findings in `docs/findings.md`, and pitfalls in `docs/gotchas.md`, record speedup between scalar and RVV kernels in `docs/speedup.md`, summarize the operators that have not yet been optimized in `docs/unoptimized.md` (remove operators once optimized)
- Every new/modified RVV kernel must have:
  - Corresponding accuracy tests (compared with scalar)
  - Corresponding benchmark (compared with scalar)
  - Commit message explaining the operator, VLEN assumptions, and test conditions

## Don'ts

- Do not modify the logic of upstream LiteRT's existing ARM Neon code (read-only, use as a reference)
- Do not mix schema's push/migrate modes (not an issue in this project, but stay alert)
- Do not expose in PR titles/descriptions/public docs: Intranet IPs, personal info, unredacted customer names (refer to rule 6 in the global CLAUDE.md)

## Task Completion Format

According to rule 8 in the global CLAUDE.md, provide the following at the end of each task:
```
Changes: ...
Validation: ...
Pending: ...
```
