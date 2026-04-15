# EVolvAI — Module: Geospatial Dashboard & API

This module provides the interactive visualization layer for the EVolvAI scenario planning framework. It maps the IEEE 33-bus system onto neighborhoods in **New York City (Manhattan & Brooklyn)**.

## 🚀 Key Features
- **NYC Grid Mapping**: Real GPS coordinates for all 33 IEEE bus nodes distributed across Manhattan neighborhoods.
- **Model-In-The-Loop**: The API dynamically loads generated `.npy` demand tensors from the `generative_core` output.
- **Folium Visualization**: Heatmaps and status markers driven by real generative AI outputs.
- **OpenChargeMap Integration**: Comparison of simulated demand against real-world NYC charging infrastructure.

## 🚦 How to Run

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. Start the API (Terminal 1)
```bash
python -m uvicorn api:app --reload
```

### 3. Start the Dashboard (Terminal 2)
```bash
streamlit run dashboard.py
```

## 🔌 API Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/api/nodes` | Baseline node configuration in NYC |
| GET | `/api/nodes/{scenario}` | Nodes adjusted with model-generated demand |
| GET | `/api/gini/{scenario}` | Equity statistics (Gini index) for a scenario |
| GET | `/api/scenarios` | List of supported counterfactuals |

## 📉 Supported Scenarios
The dashboard supports all scenarios defined in the `generative_core` config:
- `baseline`
- `extreme_winter_storm`
- `summer_peak`
- `full_electrification`
- `extreme_winter_v2`
- `rush_hour_gridlock`

> [!NOTE]
> If a specific `.npy` file is missing for a scenario in the `output/` directory, the API will automatically fall back to a physics-proxy multiplier logic.