# HPGR Performance Prediction System

An offline, physics-informed machine learning decision-support simulator and interactive Streamlit dashboard built from the 21 pilot-scale High Pressure Grinding Roll (HPGR) tests in **Jeffrey Adam Drozdiak (UBC, 2011)** — *A Pilot-Scale Examination of a Novel High Pressure Grinding Roll / Stirred Mill Comminution Circuit for Hard-Rock Mining Applications*.

> **Important Scope & Limitations**
> - **Source material:** Pilot-scale **Mesaba copper-nickel sulphide ore** (`D = 0.750 m`, `L = 0.220 m`, static gap `9.0 mm`). This is **not** iron-ore plant data.
> - **Small sample size:** Only **21 real test records** (`n = 21` for throughput & specific energy; `n = 16` for product `P80` targets). Suitable for prototype model comparison via Leave-One-Out Cross-Validation (LOOCV), not industrial-grade guarantees.
> - **Equipment boundary:** Not validated for Extract Met equipment or full-scale industrial HPGR circuits.
> - **Offline prototype:** No PLC/SCADA connection, automated control, or 3D digital twin.
> - **Bond Work Index (BWi) / Micro-cracking:** Not included in `HPGR_Extracted_Data.xlsx` and **not fabricated**; clearly marked as future work.

---

## Project Structure

```text
├── app.py                              # Home page & workflow overview
├── requirements.txt                    # Python dependencies
├── README.md
├── .streamlit/config.toml              # Light colorful theme configuration
├── config/
│   └── physics_assumptions.yaml        # Machine geometry, feature/target specs, synthetic policy
├── data/
│   ├── real/
│   │   └── HPGR_Extracted_Data.xlsx    # Preserved source workbook (21 real pilot tests)
│   ├── processed/
│   │   ├── hpgr_validated_data.xlsx    # Validated copy + Data_Dictionary + Physics_Audit
│   │   ├── data_dictionary.csv
│   │   ├── model_evaluation_report.csv
│   │   └── loocv_predictions.csv
│   └── synthetic/
│       └── hpgr_synthetic_exploration.csv  # 40 synthetic rows (exploration only, never mixed)
├── models/
│   ├── metadata.json                   # Feature bounds, LOOCV metrics, kernel/coef summaries
│   └── *.joblib                        # 15 saved pipelines (5 targets × 3 model families)
├── pages/
│   ├── 1_Data_Overview.py              # Workbook inspection, upload, data dictionary, physics audit
│   ├── 2_Exploratory_Analysis.py       # Interactive EDA scatter plots, correlations & feasibility
│   ├── 3_Prediction.py                 # Real-time predictions, presets, GPR ±1.96σ & warnings
│   ├── 4_Model_Comparison.py           # LOOCV metrics table, parity & residual plots
│   ├── 5_Scenario_Analysis.py          # Baseline vs. modified what-if comparison & 1D sweeps
│   └── 6_Confidence.py                 # Training range audit, GPR σ profile & validation roadmap
├── src/
│   ├── data_loader.py
│   ├── data_validation.py
│   ├── preprocessing.py
│   ├── physics/
│   │   ├── hpgr_equations.py
│   │   └── feature_engineering.py
│   ├── models/
│   │   ├── train_models.py
│   │   ├── predict.py
│   │   └── uncertainty.py
│   ├── evaluation/
│   │   └── metrics.py
│   └── visualization/
│       ├── ui_theme.py
│       └── charts.py
└── tests/
    ├── test_data_loader.py
    ├── test_physics_equations.py
    └── test_models_and_uncertainty.py
```

---

## Setup & Execution

```bash
pip install -r requirements.txt
python -m src.models.train_models
pytest -v
streamlit run app.py
```

---

## Physics Equations Implemented (`D = 0.750 m`, `L = 0.220 m`)

Verified directly against Appendix C (Table C-2) and `Operating_Data`:
- **Specific Pressing Force:** $F_{sp} = \frac{F}{D \cdot L \cdot 1000}\;\text{[N/mm}^2\text{]}$
- **Specific Throughput Constant:** $\dot{m} = \frac{W}{D \cdot L \cdot v}\;\text{[t}\cdot\text{s/(m}^3\cdot\text{h)]}$
- **Peripheral Roll Speed:** $v = \frac{\pi \cdot D \cdot \text{rpm}}{60}\;\text{[m/s]}$
- **Net Specific Energy (preferred):** $E_{sp,\text{net}} = \frac{P_t - P_i}{W}\;\text{[kWh/t]}$
- **Centre vs. Edge Product Size:** Modeled separately (`centre_P80_mm`, `edge_P80_mm`, `scaleup_P80_mm`) without imposing an unverified weighting.

---

## Model Evaluation Summary (Leave-One-Out Cross-Validation)

All preprocessing (`SimpleImputer`, `StandardScaler`) runs inside each `Pipeline` fold with zero data leakage.

| Target | Unit | Usable Rows (`n`) | Linear Regression R² / RMSE | Random Forest R² / RMSE | GPR (DotProduct + RBF) R² / RMSE |
|---|---:|---:|---|---|---|
| `throughput_t_h` | t/h | 21 | 0.3535 / 2.8822 | -0.4715 / 4.3481 | **0.5311 / 2.4544** |
| `net_specific_energy_kWh_t` | kWh/t | 21 | 0.8313 / 0.1089 | 0.1765 / 0.2407 | **0.8515 / 0.1022** |
| `scaleup_P80_mm` | mm | 16 | **0.9250 / 0.6427** | 0.8119 / 1.0182 | 0.9179 / 0.6728 |
| `centre_P80_mm` | mm | 16 | **0.8870 / 0.7407** | 0.8107 / 0.9586 | 0.8753 / 0.7781 |
| `edge_P80_mm` | mm | 16 | 0.9193 / 0.9468 | 0.7407 / 1.6967 | **0.9201 / 0.9421** |

---

## Additional Plant Measurements Needed for Future Validation

1. **Paired Feed & HPGR Product Bond Work Index (`BWi`)** to quantify downstream micro-cracking grindability benefit.
2. **Ore Competence / SMC / Drop-Weight Parameters (`A × b`)** for ore types beyond Mesaba Cu-Ni sulphide ore.
3. **Complete Bulk Density & Moisture Measurements** (`feed_bulk_density_t_m3` is missing in 6 of 21 pilot records).
4. **Roll Surface / Stud Wear & Edge-Recirculation Tracking** across industrial operating campaigns.
5. **Calibrated Floating Roll-Gap Force Balance Instrumentation** to model `actual_roll_gap_mm` physically.
6. **Full-Scale Plant Validation Trials** across commercial roll diameters, widths, and closed-circuit screen configurations.
