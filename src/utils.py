import numpy as np
import pandas as pd

def make_train_test_split(data, holdout_ratio=0.1, seed=1234, verbose=False):
    np.random.seed(seed)
    train = data.copy()
    test = pd.DataFrame(0, index=data.index, columns=data.columns)

    for user in range(data.shape[0]):
        liked_idx = np.where(data.iloc[user] == 1)[0]
        test_size = int(np.floor(len(liked_idx) * holdout_ratio))

        if test_size > 0:
            test_idx = np.random.choice(liked_idx, size=test_size, replace=False)
            train.iloc[user, test_idx] = 0
            test.iloc[user, test_idx] = 1

    held_out_counts = test.sum(axis=1)
    print("Held-out per user (summary):")
    print(held_out_counts.describe())

    if verbose:
        for u in range(min(100, len(data))):
            print(f"User {u}: held-out = {held_out_counts.iloc[u]}")

    return train, test
