# PCEPrediction

Power Conversion Efficiency (PCE) prediction for solar cells using machine learning.

## Project Structure

```bash
PCEPrediction/
├── code/
│   ├── __init__.py
│   ├── config.py          # Configuration parameters
│   ├── data_utils.py      # Data loading and cleaning
│   ├── models.py          # Model definitions (MLP, etc.)
│   ├── training.py        # Training and evaluation logic
│   └── visualization.py   # Visualization functions
├── run_mlp_pipeline.py    # Main pipeline script
├── pce_data_analysis.ipynb # Exploratory data analysis
└── notes.md               # Technical notes and tutorials
```

## Modules Summary

| Module | Description |
|---|---|
| **Data Loading** | Load data from CSV, separate features and targets. |
| **Data Cleaning** | Handle missing values, variance filtering, correlation filtering, and scaling. |
| **PCA** | Dimensionality reduction using Principal Component Analysis. |
| **Models** | PyTorch MLP Regressor and other model definitions. |
| **Training** | Model training, cross-validation, and hyperparameter tuning. |
| **Visualization** | Plotting training history, predictions vs actual, and feature importance. |

## Usage

To run the full MLP pipeline:

```bash
python run_mlp_pipeline.py
```

## Environment

Requires:
- Python 3.x
- pandas
- scikit-learn
- torch
- matplotlib
- seaborn
