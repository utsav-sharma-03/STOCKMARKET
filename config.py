from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent
DATA_FILE_CANDIDATES = [
    ROOT_DIR / "news_features.csv",
    ROOT_DIR / "data" / "dataset.csv",
]
MODEL_DIR = ROOT_DIR / "models"
MODEL_PATH = MODEL_DIR / "best_model.pkl"
MODEL_META_PATH = MODEL_DIR / "model_metadata.json"
