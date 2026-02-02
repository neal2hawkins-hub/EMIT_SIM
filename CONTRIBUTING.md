# Contributing

Thanks for your interest in EMIT.

## Scope and expectations
- EMIT is a **reproducible computational toy model**. Please keep claims and language aligned with what the logged results support.
- If you propose physics interpretations, keep them clearly labeled as hypotheses.

## How to contribute
1. Fork the repo and create a feature branch.
2. Keep changes small and testable.
3. Add/refresh tests under the `analysis/` tooling when appropriate.
4. Open a Pull Request describing:
   - What changed
   - Why it changed
   - How to validate (commands + expected outputs)

## Reproducibility rules
- Any change that affects results must include:
  - the run configuration (`config.json`), and
  - regenerated figures/tables (or the script output that regenerates them).

## Code style
- Prefer clear, explicit code over cleverness.
- Avoid adding heavy dependencies unless necessary.
