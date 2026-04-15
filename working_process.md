# EVolvAI: End-to-End Operational Guide (Updated)

This document provides a step-by-step process for running the EVolvAI generative demand and risk engine pipeline. The project is now fully integrated and works end-to-end.

## 🚀 Step-by-Step Running Process

To run the project end-to-end using the latest integrated logic:

### 1. Environment Setup
Ensure you are using the provided virtual environment:
```bash
source venv/bin/activate
```

### 2. Generate Base Scenarios
Initialize the output directory and generate counterfactual demand profiles:
```bash
python3 run.py mock
python3 run.py generate
```
> [!NOTE]
> If a model-architecture mismatch is detected, the system will automatically fall back to pre-existing scenarios in the `output/` folder.

### 3. Run Genetic Algorithm (Risk Engine)
Find the optimal placement of chargers across the 32 load nodes:
```bash
python3 run.py optimize
```
- **Output:** `output/final_optimal_layout.json` (Optimized port counts).

### 4. Start the Geospatial Dashboard
Launch the visualization layer:
```bash
# In one terminal
cd geospatial_dashboard
uvicorn api:app --port 8000 --reload

# In another terminal
cd geospatial_dashboard
streamlit run dashboard.py --server.port 8501
```

---

## ⚡ Integration Features

### 1. Optimal Layout Visualization
The dashboard now includes a **"Show Recommended EV Chargers (Risk Engine)"** toggle in the sidebar. When enabled:
- **Gold Rings:** Yellow outlines appear on the map to indicate GA-recommended charger placements.
- **Port Counts:** Tooltips show the recommended number of fast-charging ports for each node.

### 2. Scenario Comparison
Switch between scenarios (e.g., *Extreme Winter Storm*, *Peak Summer*) to see how infrastructure recommendations hold up against different demand extremes.

---

## ✅ Final Verification Status
- **Pipeline:** End-to-end execution (Mock -> Generate -> Optimize -> Visualize) is confirmed functional.
- **GA Results:** Optimization produces non-zero results stored for UI consumption.
- **Integration:** The backend API and frontend Dashboard are successfully bridged.
