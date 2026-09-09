from pathlib import Path

class AnalysisPaths:

    def __init__(self, root):
        root = Path(root)
        self.results = root
        self.models = root / "models"
        self.analysis = root / "analysis"
        self.processed = Path("../../data/processed")
        self.benchmark_csv = root / "benchmark.csv"

    def dataset(self, method, latent):
        return self.processed / f"larry_{method}_{latent}.h5ad"

    def checkpoint(self, method, latent, hidden, seed):
        return (
            self.models
            / method
            / f"latent_{latent}"
            / f"hidden_{hidden}"
            / f"seed_{seed}"
            / "best_model.pt"
        )