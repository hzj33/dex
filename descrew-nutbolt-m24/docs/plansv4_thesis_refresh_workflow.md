# PLANS_v4 Thesis Refresh Workflow

This workflow keeps PLANS_v4 thesis artifacts synchronized with log-backed evidence.

## One-Command Refresh

```bash
python scripts/build_plansv4_thesis_bundle.py
```

## Regenerated Outputs

- `docs/data/plansv4_thesis_candidate_table.csv`
- `docs/data/plansv4_thesis_delta_table.csv`
- `docs/data/plansv4_thesis_tables.tex`
- `docs/plansv4_thesis_result_bundle.md`
- `docs/plansv4_thesis_results_subsection_draft.md`

## Suggested Writing Flow

1. Refresh artifacts with the command above.
2. Use `docs/plansv4_thesis_results_subsection_final.md` as the base paragraph in the thesis main text.
3. Use `docs/plansv4_thesis_appendix_integration.md` to insert the appendix table and references.
4. Do not hand-edit numeric values in prose; regenerate from logs when numbers need updates.

