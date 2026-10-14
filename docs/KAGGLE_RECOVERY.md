# Kaggle bounded recovery run

The first real-data pilot did not reach training. It built all 1,024 records,
then hung while the Hugging Face/PyArrow process was shutting down. Kaggle
eventually cancelled the run. The saved status proves that only environment
setup and tests returned successfully.

The recovery kernel addresses that failure directly:

- exits the completed data-builder process without waiting on library cleanup
  threads;
- admits 64 records per domain;
- trains at a 512-token maximum sequence length;
- bounds every expert to two optimizer steps;
- saves and evaluates at every optimizer step;
- copies artifacts after the router and after each two-expert GPU wave.

The recovery package is configured for the collaborator account
`bhabhihunter`. The account owner must create the source dataset and launch the
kernel themselves while authenticated as that account:

```bash
kaggle datasets create -p kaggle/dataset
kaggle kernels push -p kaggle/recovery
```

Do not share a Kaggle password, API token, `kaggle.json`, or browser session.
