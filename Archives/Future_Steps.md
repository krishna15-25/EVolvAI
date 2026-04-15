# Future Steps

> **Last Updated**: 2026-04-15 | **Geographic Focus**: New York City, NY (full pivot from California/Caltech)

1. **Test E2E Scenarios with Dashboard UI**: Launch the streamlit dashboard to verify that visual representations (Nodes, Gini Score maps) reflect `output/final_optimal_layout.json`.
2. **Refine Subsystem Interfaces**: Ensure the shapes of demand tensors seamlessly cross the boundary from PyTorch Generation -> Risk GA -> Geospatial representation. If padding is required, consider re-orienting the datasets to conform exactly to IEEE 33-Bus configuration explicitly.
3. **Hardware Acceleration**: Implement tensorized versions of the physics constraints in PyTorch for use on GPUs directly under GA to speed up evaluating thousands of risk variants.
4. **Data Contract Compliance**: Update `ROADMAP.md` checklist marks for "Integration" since they are now implemented.
5. **[COMPLETED] Colab Training Notebook (Lochan Integration)**: Rebuilt `EVolvAI_Training.ipynb` as a fully self-contained Colab notebook. It clones the repo, runs a pure-Python port of Lochan's `GenerateRandomSchedule_new_ScenerioGenerator.m` MATLAB EV scenario generator, builds the entire dataset in RAM, trains the physics-informed TCN-VAE, and downloads results. No MATLAB license required.
6. **[COMPLETED] Integrate NYC Traffic Flow Data**:
    - ✅ **Source**: NYC Open Data ATVC (`Automated_Traffic_Volume_Counts_20260415.csv`, 436k rows, 2021–2026)
    - ✅ **Schema parsed**: `Yr, M, D, HH, MM, Vol` — aggregated to hourly volumetric index per day
    - ✅ **Per-day profiles** saved to `data/processed/traffic_daily_24h.parquet` for day-level accuracy in the dataset builder
    - ✅ **Canonical 24h profile** saved to `data/processed/traffic_tensor_24h.npy` as fallback
    - ✅ **NYC-tuned FHWA fallback** profile coded (pronounced 8am/6pm Manhattan rush)
7. **[COMPLETED] NYC Geographic Pivot**:
    - ✅ All California/Caltech/PeMS/Pasadena references purged from `Latest_Training.txt`
    - ✅ Weather source: Open-Meteo Historical API → NYC (lat=40.7128, lon=-74.0060), 2021-07-31 → 2026-04-13
    - ✅ EV Charging source: NYC PlugNYC Open Data (233,865 sessions, `Date, Connected Time, Disconnected Time, Energy Provided (kWh), Charge Duration (min)`)
    - ✅ `CFG` dataclass now includes `CITY`, `LAT`, `LON`, `DATA_START`, `DATA_END` constants
8. **[NEXT] Run Full 200-Epoch Training on Colab T4**:
    - Upload repo to Google Drive → open `Latest_Training.txt` (rename to `.ipynb`) → Runtime: T4 GPU
    - Expected: ~8–12 hrs for 200 epochs on 5,000 samples
    - Monitor: R² should cross 0.9 by epoch 80; physics penalties should plateau by epoch 120
13. **Algorithm Benchmarking**: Evaluate **Particle Swarm Optimization (PSO)** against the current Genetic Algorithm (GA) for placement optimization to see if "Global Empirical Range Embedding" (GERE) concepts improve convergence speed.
14. **Refine Output Bounding**: Implement hard-clamping logic in the generation script to complement the soft penalties in `physics_loss.py`, ensuring 100% of generated scenarios are physically bounded before handoff.
15. **[UPDATED] Train the Gen-Core (GCD-VAE)**: NYC data is now the sole training source — 5,000 bootstrapped NYC scenarios bootstrapped from 233,865 real sessions × 32 IEEE-33 nodes, weighted by NYC ATVC hourly traffic profile. Run `Latest_Training.txt` (rename to `.ipynb`) on Colab T4.
