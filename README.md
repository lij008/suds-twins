# SuDS Water-Level Digital Twin (PoC)

This PoC turns MAP16 water-level monitor data into a simple "digital twin" view:
- **Twin State**: latest level and rate-of-rise per sensor
- **Events**: storm-like level events (start/end/peak/severity)
- **Health**: data completeness + flatline/spike checks

## 1) Put your CSV
Copy your MAP16 exported CSV into:
`data/map16_sensor.csv`

If your CSV headers differ, edit `twin_config.json` -> `columns_hint`.

## 2) Install
```bash
pip install -r requirements.txt
```

## 3) Generate outputs
```bash
python process_data.py
```

Outputs will be written to `outputs/`.

## 4) Run dashboard
```bash
streamlit run app.py
```

## Notes
- Times are parsed as UTC.
- Thresholds in `twin_config.json` are starting defaults; tune to your site.
