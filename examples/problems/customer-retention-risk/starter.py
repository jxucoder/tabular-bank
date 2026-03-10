from __future__ import annotations

import pandas as pd
from sklearn.ensemble import RandomForestClassifier


TRAIN_PATH = "train.csv"
TEST_PATH = "test.csv"
OUTPUT_PATH = "submission.csv"
ID_COL = "row_id"
TARGET_COL = "will_churn"


def _encode(train_df: pd.DataFrame, test_df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    train_enc = pd.get_dummies(train_df, dummy_na=True)
    test_enc = pd.get_dummies(test_df, dummy_na=True)
    train_enc, test_enc = train_enc.align(test_enc, join="outer", axis=1, fill_value=0)
    return train_enc, test_enc


train = pd.read_csv(TRAIN_PATH)
test = pd.read_csv(TEST_PATH)

X_train = train.drop(columns=[ID_COL, TARGET_COL])
y_train = train[TARGET_COL]
X_test = test.drop(columns=[ID_COL])
X_train, X_test = _encode(X_train, X_test)

model = RandomForestClassifier(n_estimators=200, random_state=0)
model.fit(X_train, y_train)

pred = model.predict_proba(X_test)[:, 1]
submission = test[[ID_COL]].copy()
submission[TARGET_COL] = pred
submission.to_csv(OUTPUT_PATH, index=False)
print(f"Wrote {OUTPUT_PATH}")
