# California House Price Prediction

Production-minded machine learning project for predicting California median house values from district-level housing features.

The original exploratory notebook was converted into a reproducible Python project with a training pipeline, saved model artifacts, evaluation metrics, and a Streamlit inference app.

## Project Structure

```text
.
├── app.py                 # Streamlit app for interactive predictions
├── train_model.py         # Training, evaluation, and artifact export
├── house_price_prediction.ipynb
├── housing.csv            # Source dataset
├── requirements.txt       # Runtime dependencies
├── .gitignore
└── artifacts/             # Generated locally: model + metrics
```

## Model

- Algorithm: `RandomForestRegressor`
- Preprocessing:
  - median imputation for numeric columns
  - standard scaling for numeric columns
  - one-hot encoding for `ocean_proximity`
- Validation:
  - stratified train/test split based on median income bands
  - grid search with cross-validation

## Run Locally

Create and activate a virtual environment, then install dependencies:

```bash
pip install -r requirements.txt
```

Train the model:

```bash
python train_model.py
```

Run the app:

```bash
streamlit run app.py
```

The first app run also trains the model automatically if `artifacts/house_price_model.joblib` does not exist.

## Deployment

The simplest deployment target is Streamlit Community Cloud:

1. Push this repository to GitHub.
2. Create a new Streamlit app from the repository.
3. Set `app.py` as the entrypoint.
4. Let Streamlit install dependencies from `requirements.txt`.

Generated model artifacts are ignored by git. For cloud deployment, the app trains the model during its first startup.
