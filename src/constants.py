from os.path import abspath, dirname, join
from pathlib import Path

project_dir = Path(abspath(join(dirname(__file__), ".."))).resolve()
data_dir = project_dir / "data"
raw_data_dir = data_dir / "raw"
processed_data_dir = data_dir / "processed"
model_data_dir = data_dir / "model"
pred_data_dir = data_dir / "pred"
models_dir = project_dir / "models"
reports_dir = project_dir / "reports"

raw_data_name = "FactorTimingData-ABVU2022.xlsx"
