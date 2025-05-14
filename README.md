# Molecular Solubility Prediction

This repository contains a comprehensive implementation of a transformer-based model for predicting molecular solubility from SMILES representations.

## Project Overview

Molecular solubility prediction is a critical task in drug discovery and materials science. This project implements a state-of-the-art approach using transformer models with a fusion of SMILES representations and molecular descriptors to achieve high accuracy predictions.

## Repository Structure

```
solubility-prediction/
├── Project_Artifacts/
│   ├── Data_Exploration/    # Data exploration code and visualizations
│   └── ML_Code/             # ML implementation for solubility prediction
├── Project_Input/           # Input data including ESOL dataset
├── Project_Memory/          # Project documentation and reports
└── Project_To_Do_List/      # Task descriptions and planning documents
```

## Key Features

- **High Performance**: Test RMSE of 0.778 log units and R² of 0.863
- **Transformer Architecture**: 6-layer transformer with token-level fusion
- **Comprehensive Pipeline**: From data exploration to model deployment
- **Extensive Documentation**: Complete analysis and implementation details

## Performance Highlights

| Metric | Value | Target | Status |
|--------|-------|--------|--------|
| RMSE   | 0.778 | <1.0   | ✅     |
| R²     | 0.863 | >0.7   | ✅     |
| MAE    | 0.592 | <0.7   | ✅     |

## Quick Start

The main implementation is in the ML_Code directory. See [ML_Code/README.md](Project_Artifacts/ML_Code/README.md) for detailed usage instructions.

To train the model:
```bash
cd Project_Artifacts/ML_Code
python main.py --mode train --data_path ../../Project_Input/solubility_esol.csv --smiles_col smiles --solubility_col solubility
```

## Documentation

- [Data Exploration Results](Project_Memory/data_exploration_results.txt)
- [Literature Search Results](Project_Memory/literature_search_results.txt)
- [ML Repository Search Results](Project_Memory/ml_repo_search_results.txt)
- [Transformer Model Plan](Project_Memory/transformer_model_plan.txt)
- [Data Preprocessing Plan](Project_Memory/data_preprocessing_plan.txt)
- [Evaluation Plan](Project_Memory/evaluation_plan.txt)
- [ML Execution Report](Project_Memory/ml_execution_report.txt)

## License

This project is licensed under the MIT License - see the LICENSE file for details.
