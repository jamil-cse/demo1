# Diabetes Risk Predictor: ML model deployed with Streamlit

A complete small project: train a model on real data, save it, and serve it as a Streamlit web app.

| File | Purpose |
|---|---|
| `train_model.py` | downloads the Pima Indians Diabetes data, compares 3 models, saves `model.joblib` and `metrics.json` |
| `app.py` | Streamlit app: prediction form, what-if curve, model performance, data notes |
| `model.joblib`, `metrics.json` | the trained model and its metrics (already included, ready to deploy) |
| `requirements.txt` | dependencies (scikit-learn is pinned to the version the model was trained with) |

## Run locally
```
pip install -r requirements.txt
python train_model.py        # optional: retrain (the model files are already included)
streamlit run app.py
```

## Deploy free on Streamlit Community Cloud
1. Push this folder to a new GitHub repository (include `model.joblib` and `metrics.json`).
2. Go to share.streamlit.io, sign in with GitHub, click **Create app**, choose the repository, branch `main`, main file `app.py`.
3. Click **Deploy**. You get a public link in a couple of minutes.

If you retrain with a different scikit-learn version, update the version pinned in `requirements.txt`.

Educational demo only. Not a medical device.
