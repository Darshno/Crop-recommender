# Phase 2 independent-data audit

Run from the project root:

```powershell
.venv\Scripts\python.exe experiments\independent_data_v2_audit\phase2_audit.py
```

This phase is descriptive and read-only. It performs no model fitting, tuning,
or artifact replacement. Outputs include shared-class statistics, feature-label
relationship analysis, cross-source nearest-neighbor diagnostics, plots,
unresolved data questions, and a leakage-free evaluation protocol.
