# PLANS_v4 Thesis Appendix Integration

This note provides copy-ready snippets to integrate PLANS_v4 evidence tables into a thesis manuscript.

## 1. Main-Text Reference Snippet

Use this in the results section to reference the appendix table:

```tex
As summarized in Table~\ref{tab:plansv4-candidate-gate}, none of the V3/V4 diffusion candidates
passed the unified single-seed gate, and the bug-free V4 baseline still triggered the hard-stop
condition ($\Delta_{\text{hard}}=-0.137635$).
```

## 2. Appendix Table Wrapper (Recommended)

Wrap the generated table body (`docs/data/plansv4_thesis_tables.tex`) with a standard table environment:

```tex
\begin{table}[t]
  \centering
  \caption{PLANS\_v4 candidate summary under the unified single-seed gate (seed=42, steps=256).}
  \label{tab:plansv4-candidate-gate}
  \input{docs/data/plansv4_thesis_tables.tex}
\end{table}
```

## 3. Optional CSV-Based Figure/Table Pipelines

If your thesis uses external plotting/table tooling:

- candidate full table: `docs/data/plansv4_thesis_candidate_table.csv`
- compact delta table: `docs/data/plansv4_thesis_delta_table.csv`

## 4. Source of Truth

Always treat these as authoritative in descending order:

1. `docs/plansv4_m3_final_verdict.md`
2. `docs/plansv4_thesis_result_bundle.md`
3. `docs/data/plansv4_thesis_candidate_table.csv` and `docs/data/plansv4_thesis_delta_table.csv`

