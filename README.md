# Hotel Clustering Intelligence Dashboard

This project uses a saved K-Means model and related artifacts to power an interactive Streamlit dashboard for hotel segmentation.

## Project Structure

- `data/Hotel_Reviews.csv` - source dataset
- `models/final_hotel_clustering_model.pkl` - saved K-Means model
- `models/hotel_scaler.pkl` - saved StandardScaler
- `models/hotel_features.pkl` - saved feature list
- `models/hotel_pca.pkl` - saved PCA object used for visualization
- `results/hotel_clusters.csv` - clustered hotel data
- `results/cluster_profile.csv` - summary of cluster metrics
- `results/clustering_algorithm_comparison.csv` - algorithm comparison metrics
- `app.py` - Streamlit application

## Run the Dashboard

Create and activate a virtual environment:

```bash
python -m venv .venv
source .venv/bin/activate
```

On Windows PowerShell:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

Install dependencies:

```bash
pip install -r requirements.txt
```

Run the dashboard:

```bash
streamlit run app.py
```

The dashboard will open in the browser.

## Notes

- The dashboard loads the already-trained K-Means model and scaler from the `models/` folder.
- It does not retrain the model during startup or prediction.
- PCA is kept for visualization only and is not used in prediction.
- Latitude and longitude are used only for the geographic view and not for model training or prediction.
- The app includes CSV upload validation for empty files, missing columns, and invalid numeric values.
