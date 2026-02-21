"""
Optuna Diagnostics Visualizer.
Reads an Optuna SQLite database and generates interactive Plotly HTMLs
for hyperparameter importances (F-ANOVA) and parallel coordinates.
"""

from pathlib import Path
import optuna
from optuna.visualization import plot_param_importances, plot_parallel_coordinate


def generate_optuna_diagnostics(db_path: str, study_name: str, out_dir: str):
    """
    Generate interactive HTML diagnostics for a given Optuna study.

    Args:
        db_path: Path to the SQLite database file (e.g., 'results/optuna/earthquake_pinn_hpo.db').
        study_name: Name of the study to load.
        out_dir: Directory to save the generated HTML files.
    """
    db_file = Path(db_path)
    if not db_file.exists():
        print(f"Error: Database file missing at {db_path}.")
        return

    out_path = Path(out_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    storage_url = f"sqlite:///{db_file.absolute()}"

    try:
        study = optuna.load_study(study_name=study_name, storage=storage_url)
        print(f"Loaded study '{study_name}' with {len(study.trials)} trials.")
    except Exception as e:
        print(f"Failed to load study '{study_name}': {e}")
        return

    if len(study.trials) == 0:
        print("No trials found. Skipping Optuna diagnostics.")
        return

    try:
        # F-ANOVA Param Importances
        fig_importances = plot_param_importances(study)
        importances_file = out_path / "optuna_param_importances.html"
        fig_importances.write_html(str(importances_file))
        print(f"Generated {importances_file}")
    except Exception as e:
        print(f"Failed to plot parameter importances: {e}")

    try:
        # Parallel Coordinates
        fig_parallel = plot_parallel_coordinate(study)
        parallel_file = out_path / "optuna_parallel_coordinates.html"
        fig_parallel.write_html(str(parallel_file))
        print(f"Generated {parallel_file}")
    except Exception as e:
        print(f"Failed to plot parallel coordinates: {e}")


if __name__ == "__main__":
    generate_optuna_diagnostics(
        db_path="results/optuna/earthquake_pinn_hpo.db",
        study_name="earthquake_pinn_hpo",
        out_dir="results/figs",
    )
