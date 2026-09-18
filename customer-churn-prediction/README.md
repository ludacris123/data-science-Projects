# Customer Churn Prediction

An end-to-end binary classification project that predicts whether a subscription customer is likely to leave.

## Skills demonstrated

- Synthetic dataset generation
- Feature preprocessing with a scikit-learn pipeline
- Logistic regression and random forest comparison
- Cross-validation and holdout evaluation
- Confusion matrix, ROC-AUC, and feature importance
- Model persistence for later API integration

## Run

```bash
pip install pandas numpy scikit-learn matplotlib joblib
python train.py
```

The script creates `outputs/` containing the trained model, metrics, predictions, confusion matrix, ROC curve, and feature-importance chart.

## Production extension

Serve `churn_model.joblib` through FastAPI, validate requests with Pydantic, and monitor prediction drift.
