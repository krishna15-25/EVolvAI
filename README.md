# EVolvAI: Generative Scenario Planning for EV Demand

[![License: GPL v3](https://img.shields.io/badge/License-GPLv3-blue.svg)](https://www.gnu.org/licenses/gpl-3.0)

> **Causal Demand Forecasting via Generative Counterfactuals.**

EVolvAI is a specialized framework designed to forecast Electric Vehicle (EV) charging demand using a **Generative Causal Demand VAE (GCD-VAE)**. The system enables "Scenario Planning" by intervening on causal factors like weather, fleet size, and traffic volume to predict grid stress.

## 🚀 Geographic Scope: New York City
The framework is currently calibrated for **New York City (Manhattan & Brooklyn)**, integrating:
- **PlugNYC Data**: Extensive real-world charging session logs from NYC DOT.
- **Caltech ACN-Data**: High-granularity behavioral distributions.
- **IEEE 33-Bus Topology**: CAN-standard grid physics mapped to NYC neighborhoods.

## 🏗️ Integrated Architecture
The project follows a **Model-In-The-Loop** workflow:
1. **Data Pipeline**: Sources charging behavioral data from NYC municipal datasets.
2. **Generative Core**: Trains a TCN-VAE to learn the latent space of demand.
3. **Scenario Generator**: Produces NumPy tensors (`.npy`) for specific counterfactuals (e.g., "Extreme Winter Storm").
4. **Geospatial Dashboard**: A FastAPI + Streamlit interface that visualizes these tensors on a real-world map of the NYC grid.

## 📂 Project Structure
- `generative_core/`: GCD-VAE architecture, training, and generation logic.
- `data_pipeline/`: Modules for OSMnx road network and traffic volume processing.
- `geospatial_dashboard/`: Interactive visualization and REST API.
- `output/`: Trained checkpoints and generated scenario tensors.

## 🚦 Quick Start

### Installation
```bash
git clone https://github.com/seeramsujay/EVolvAI
cd EVolvAI
pip install -r requirements.txt
```

### Running the Full Pipeline
```bash
# Generate scenarios from the trained model
python run.py generate

# Launch Dashboard
cd geospatial_dashboard
pip install -r requirements.txt
python -m uvicorn api:app --reload &
streamlit run dashboard.py
```

## 📜 License
This project is licensed under the **GNU General Public License v3.0**. See the [LICENSE](LICENSE) file for details.
