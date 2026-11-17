"""Re-train the Pattern Classifier and regenerate reports/metrics.json.

    python train.py
"""
from seismicsense.model import train

if __name__ == "__main__":
    train(verbose=True)
