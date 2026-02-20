import numpy as np

def calculate_soc(preds):
    preds = np.array(preds, dtype=float)  # <-- convert to NumPy array
    return np.clip(100 - preds * 0.5, 0, 100)

def calculate_soh(preds):
    preds = np.array(preds, dtype=float)
    return np.clip(100 - preds * 0.2, 0, 100)

def calculate_rul(preds):
    preds = np.array(preds, dtype=float)
    return np.clip(1000 - preds * 5, 0, 1000)
