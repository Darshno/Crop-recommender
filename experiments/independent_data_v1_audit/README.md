# Independent-data v1 forensic audit

This directory contains a read-only audit of the existing experiment. It does
not change the source CSVs, frozen baseline, production application, or v1
experimental artifacts.

Run from the project root:

```powershell
.venv\Scripts\python.exe experiments\independent_data_v1_audit\audit_experiment.py
```

The report distinguishes confirmed implementation/evaluation facts from
potential distribution, unit, and provenance problems requiring verification.
