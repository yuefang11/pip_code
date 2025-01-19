import numpy as np
import pandas as pd

import pickle



data_all = pd.read_stata('dataset/application/realdata_used.dta')
data_all = data_all[data_all["prevearn"] <= 15000]
dummies = pd.get_dummies(data_all["siteno"], prefix="siteno", dtype = float)
data_all = pd.concat([data_all, dummies], axis=1)
dummy_race = pd.get_dummies(data_all["race"], prefix="race", dtype = float)
data_all = pd.concat([data_all, dummy_race], axis=1)
data_all = data_all.drop(["siteno","race"], axis = 1)
data_all=data_all[data_all["D"]==1]
data_all = data_all[data_all["hour_train"] >= 40]
data_all = data_all[data_all["hour_train"] <= 1360]
data_all["treat"] = 0

data_all.loc[(data_all["hour_train"] > 200) & (data_all["hour_train"] <= 600), "treat"] = 1
data_all.loc[(data_all["hour_train"] > 600), "treat"] = 2
data_all = data_all.drop(["hour_train","bfhrswrk"], axis = 1)

np.random.seed(42)  # For reproducibility
n_samples = len(data_all)
fold_size = 500
# Create array of shuffled indices
shuffled_indices = np.random.permutation(n_samples)

# Split into 6 folds - first 5 with 1000 samples each
fold_indices = []
for i in range(4):
    start_idx = i * fold_size
    end_idx = start_idx + fold_size
    fold_indices.append(shuffled_indices[start_idx:end_idx])

folds = []
for indices in fold_indices:
    folds.append(data_all.iloc[indices])


for i in range(len(folds)):
    # Save each fold as a pickle file
    fold_data = folds[i]
    with open(f'application/fold_{i}.pkl', 'wb') as f:
        pickle.dump(fold_data, f)

