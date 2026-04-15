# EVolvAI: Technical Architecture & Pipeline Overview

This document provides a comprehensive breakdown of the EVolvAI project for research and documentation purposes.

---

## 1. Project Objective
**EVolvAI** is a system designed for **Equitable EV Infrastructure Planning**. It combines Physics-Constrained Generative AI with a Genetic Algorithm (GA) Risk Engine to optimize charger placement on a power grid (IEEE 33-Bus) while considering climate anomalies, traffic patterns, and grid stress.

---

## 2. High-Level Pipeline Flow
The project follows a 4-step sequential pipeline orchestrated by `run.py`:

1.  **`mock`**: Initializes the handoff between the generative core and the risk engine by creating a standard demand tensor.
2.  **`train`**: Trains a **Generative Counterfactual VAE (GCD-VAE)** to learn the relationship between weather, traffic, and charging demand.
3.  **`generate`**: Uses the trained model to produce "Counterfactual Scenarios" (e.g., *What happens to demand if we have a winter storm and 3x more EVs?*).
4.  **`optimize`**: The **GA Risk Engine** takes these scenarios and finds the optimal charger counts for every node to prevent grid collapse.
5.  **`dashboard`**: A real-world visualization layer (NYC Map) for stakeholders to review the AI's recommendations.

---

## 3. Directory Structure & Key Files

### 📁 `generative_core/` (The Generative Brain)
*   **`models.py`**: Contains the **Attention-TCN VAE** architecture. It uses Temporal Convolutional Networks (TCN) to handle time-series demand.
*   **`config.py`**: The "Single Source of Truth." Defines all hyperparameters, grid dimensions (32 nodes), and scenario conditions.
*   **`train.py` / `generate.py`**: Handles the learning and inference phases of the demand generation.
*   **`mock.py`**: Generates synthetic data for testing the pipeline in isolation.

### 📁 `risk_engine/` (The Optimization Logic)
*   **`optimizer_ga.py`**: The core Genetic Algorithm.
    *   **Chromosome**: Vector of length 32 (ports per node).
    *   **Fitness Function**: A multi-objective function balancing **CapEx** (cost), **Wait Time** (user experience), **Grid Stress** (transformer health), and **Tail Risk (CVaR)**.
*   **`__init__.py`**: (Fixed) Enables the module to be imported by the central pipeline.

### 📁 `geospatial_dashboard/` (The Visualization Layer)
*   **`api.py`**: A FastAPI backend that serves grid data, scenario metrics, and the GA's "Optimal Layout."
*   **`dashboard.py`**: A Streamlit frontend that maps the IEEE 33-Bus system onto New York City GPS coordinates.
*   **`gini.py`**: Calculates the **Gini Accessibility Index** – a metric for social equity in charger distribution.

### 📁 `output/` (The Handoff Artifacts)
*   **`gcvae_model.pt`**: The trained neural network weights.
*   **`*.npy` files**: Counterfactual demand tensors (e.g., `extreme_winter_storm.npy`).
*   **`final_optimal_layout.json`**: The final design produced by the GA.

---

## 4. Dataset Sources & Features

1.  **Power Grid**: IEEE 33-Bus standard topology (33 nodes, 32 load nodes).
2.  **Charging Data**: Based on **ACN-Data** (Caltech) charging sessions.
3.  **Weather**: Open-Meteo API (Historical temperature and precipitation).
4.  **Traffic**: Synthetic traffic index mapped to rush-hour windows.
5.  **Real Infrastructure**: **OpenChargeMap API** (Ingested live into the dashboard for "Real vs. AI" comparison).

---

## 5. Technical Interlinks (How it connects)

*   **Logic Link**: `config.py` defines a scenario → `generate.py` creates the demand for it → `optimizer_ga.py` reads that demand to calculate grid risk → `api.py` serves that final plan.
*   **Data Link**: All modules communicate via the `output/` directory using `.npy` (NumPy) and `.json` files. This "Asset-Based" handoff allows the Generative Core (PyTorch) and the Dashboard (Streamlit/FastAPI) to run independently.

---

## 6. Execution Commands (For Reproducibility)

```bash
# 1. Pipeline Run
python3 run.py all

# 2. Start Services
cd geospatial_dashboard
uvicorn api:app --port 8000 &
streamlit run dashboard.py --server.port 8501
```
