# FUTURE_STEPS.md – Making This Pipeline Production-Ready

This document is your personal checklist for transitioning the Generative Counterfactual Framework from simulated data to real-world accuracy.

---

## Step 1: Acquire Real EV Charging Data

The single most impactful upgrade. Replace the random simulation in `data_loader.py` with actual historical charging sessions.

### Where to Get It
| Source | URL | What You Get |
| :--- | :--- | :--- |
| **Caltech ACN-Data** | [ev.caltech.edu/dataset](https://ev.caltech.edu/dataset) | Timestamped sessions: start, end, energy (kWh), peak power (kW), site ID. |
| **NREL EVI-Pro** | [nrel.gov/transportation](https://www.nrel.gov/transportation/evi-pro.html) | Modeled charging load profiles for different adoption scenarios. |
| **UK Power Networks** | [ukpowernetworks.co.uk/open-data](https://www.ukpowernetworks.co.uk/open-data) | Real grid-connected EV charger data with timestamps. |
| **Boulder Open Data** | [open-data.bouldercolorado.gov](https://open-data.bouldercolorado.gov) | Public EV charging station usage logs. |

### What Format You Need
Create a `.csv` or `.parquet` file with **one row per hour** structured like this:

```csv
date,hour,node_id,demand_kw,temperature_c,precipitation_mm
2024-01-15,0,node_001,45.2,-3.1,0.0
2024-01-15,0,node_002,12.8,-3.1,0.0
2024-01-15,1,node_001,38.7,-4.2,0.5
...
```

### Where to Put It
```
EVolvAI/
└── data/
    ├── raw/                    # Original downloaded files (untouched)
    │   ├── acn_sessions.csv
    │   └── weather_boulder.csv
    └── processed/              # Cleaned, aligned, ready-to-train
        └── train_data.parquet
```

### How to Wire It In
In `generative_model/data_loader.py`, replace the `__init__` method of `EVDemandDataset`:

```python
# CURRENT (simulated):
base = np.random.uniform(10, 100, (num_samples, seq_len, num_nodes))

# REPLACE WITH:
import pandas as pd
df = pd.read_parquet('data/processed/train_data.parquet')
# Pivot so rows = (date, hour), columns = node_id, values = demand_kw
pivot = df.pivot_table(index=['date', 'hour'], columns='node_id', values='demand_kw')
# Reshape to [num_days, 24, num_nodes]
charge = pivot.values.reshape(-1, 24, num_nodes).astype(np.float32)
```

---

## Step 2: Acquire Real Weather Data

### Where to Get It
| Source | URL | Notes |
| :--- | :--- | :--- |
| **Open-Meteo** | [open-meteo.com](https://open-meteo.com) | Free, no API key. Historical hourly temp, precip, wind. |
| **NOAA ISD** | [ncdc.noaa.gov](https://www.ncdc.noaa.gov/isd) | US weather station data, hourly resolution. |
| **Visual Crossing** | [visualcrossing.com](https://www.visualcrossing.com) | Free tier, easy CSV export. |

### What You Need
Hourly data covering the **exact same date range and location** as your charging data:
- Temperature (°C)
- Precipitation (mm)
- Wind speed (optional, useful for extreme event flags)

### How to Wire It In
The weather data becomes the extra feature column(s) concatenated to the charging data in `data_loader.py`. Currently there is 1 weather column (temperature). You can add more:

```python
# In config.py, update:
NUM_WEATHER_FEATURES = 3  # temp, precip, wind
NUM_FEATURES = NUM_NODES + NUM_WEATHER_FEATURES
```

---

## Step 3: Map Nodes to Akshay's Grid Topology

### What to Ask Akshay For
A file mapping each `node_id` to his `pandapower` bus index:

```json
{
  "node_001": {"bus_index": 0, "type": "residential", "transformer_kva": 500},
  "node_002": {"bus_index": 1, "type": "commercial",  "transformer_kva": 1000},
  ...
}
```

### Where to Put It
```
EVolvAI/
└── data/
    └── grid_topology.json
```

### Why This Matters
When you train with real node identities, the AI model learns **which nodes** are residential vs. commercial and generates demand profiles that respect those physical characteristics. This is what makes the counterfactual physically realistic rather than purely statistical.

---

## Step 4: Define Better Condition Variables

Currently, the condition vector is `[WeatherFlag, EV_Multiplier]` — two floats. For richer counterfactuals, expand it:

```python
# In config.py:
COND_DIM = 5

SCENARIOS = {
    "extreme_winter_v2": {
        "description": "Winter storm + full electrification + weekend",
        "condition": [1.0, 2.5, 0.0, 1.0, 0.0],
        #             temp  ev    solar weekend holiday
    },
}
```

Suggested condition dimensions:
1. **Temperature anomaly** (float, deviation from seasonal avg)
2. **EV Electrification multiplier** (float, 1.0 = baseline)
3. **Solar generation availability** (float, 0.0 = cloudy, 1.0 = clear)
4. **Weekend flag** (0 or 1)
5. **Holiday flag** (0 or 1)

> ⚠️ When you change `COND_DIM`, the model architecture updates automatically via `config.py`. But you **must retrain** from scratch — old checkpoints won't load.

---

## Step 5: Train for Real

Once you have real data wired in:

1. Upload the full project folder to Google Drive.
2. Open `EVolvAI_Training.ipynb` in Colab.
3. Mount Google Drive and update file paths in the config cell.
4. Set `EPOCHS = 50` (or higher) and use a T4 GPU.
5. Monitor the loss — it should steadily decrease. If it plateaus early, increase `LATENT_DIM` or add more TCN layers.

### Training Tips
- **If loss explodes:** Lower `LEARNING_RATE` to `1e-4`.
- **If reconstruction is blurry:** Increase `KLD_WEIGHT` gradually (try `0.5 → 1.0 → 2.0`).
- **If the model memorizes instead of generalizing:** Add more dropout (`DROPOUT = 0.3`) or more training data.

---

## Step 6: Final Integration with Teammates

Once the model is trained on real data:

```bash
python run.py generate
```

This produces `.npy` files in `output/` for every scenario in `config.SCENARIOS`. Hand these to Akshay and Teammate B. Their code doesn't change — only the data quality improves.

---

## Checklist Summary

- [ ] Download real EV charging data → `data/raw/`
- [ ] Download matching weather data → `data/raw/`
- [ ] Write a preprocessing script → `data/processed/train_data.parquet`
- [ ] Get grid topology JSON from Akshay → `data/grid_topology.json`
- [ ] Update `data_loader.py` to read real `.parquet` files
- [ ] Expand `COND_DIM` with richer scenario triggers in `config.py`
- [ ] Train on Colab with `EPOCHS ≥ 50`
- [ ] Validate output shapes and distribute `.npy` files to team
