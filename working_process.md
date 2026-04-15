# EVolvAI: End-to-End Operational Guide

This document provides a step-by-step process for running the EVolvAI generative demand and risk engine pipeline, along with a report on issues identified during verification.

## 🚀 Step-by-Step Running Process

To run the project end-to-end without training or modification, follow these steps in the terminal from the project root:

### 1. Environment Setup
Ensure you are using the provided virtual environment:
```bash
source venv/bin/activate
```

### 2. Generate Mock Base Demand
Initialize the output directory with a standard demand tensor:
```bash
python3 run.py mock
```
- **Output:** `output/mock_demand_tensor.npy` [24 hrs, 32 nodes]

### 3. Run GA Optimization (Risk Engine)
Due to a naming typo in the package (`risk_engine/__init.py` instead of `__init__.py`), the standard entry point `run.py optimize` may fail. Use the direct script execution instead:
```bash
python3 risk_engine/optimizer_ga.py
```
- **Action:** This runs the Genetic Algorithm against current scenarios.
- **Output:** `output/final_optimal_layout.json` (Optimized port counts per node).

### 4. Start the Backend API
Navigate to the dashboard directory and launch the FastAPI server:
```bash
cd geospatial_dashboard
uvicorn api:app --port 8000 --reload
```

### 5. Launch the Dashboard
In a separate terminal, launch the Streamlit visualization:
```bash
cd geospatial_dashboard
streamlit run dashboard.py --server.port 8501
```

---

## ⚠️ Identified Issues & Workarounds

During verification, the following issues were noted:

### 1. Structural Typo
- **File:** `risk_engine/__init.py`
- **Issue:** Missing double underscores (should be `__init__.py`). This prevents `import risk_engine` from working in scripts like `run.py`.
- **Workaround:** Run `optimizer_ga.py` as a standalone script as shown in Step 3.

### 2. Generative Model Mismatch
- **File:** `generative_core/config.py` vs `output/gcvae_model.pt`
- **Issue:** The model architecture defined in the config does not match the provided checkpoint (TCN channel mismatch).
- **Workaround:** The project successfully falls back to `.npy` files already present in the `output/` folder.

### 3. Dashboard Integration Gap
- **Issue:** The `geospatial_dashboard` currently visualizes baseline counts. While the GA successfully generates `final_optimal_layout.json`, the dashboard UI is not yet configured to import and display these optimized results on the map.

### 4. Stale Services
- **Issue:** Pre-existing processes on ports 8000/8501 were using an older code version. 
- **Action:** Restarting the services as per steps 4 & 5 ensures full functionality.

---

## ✅ Validation Results
- **Pipeline:** End-to-end execution is functional using the workarounds above.
- **GA Results:** Optimization results are non-zero and stored in `output/final_optimal_layout.json`.
- **UI:** The map and scenario switching are stable and reactive.
