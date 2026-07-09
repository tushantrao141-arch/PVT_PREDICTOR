import os
import numpy as np
import pandas as pd
from sklearn.model_selection import GroupShuffleSplit
from sklearn.preprocessing import StandardScaler
import joblib


def add_engineered_features(df):
    df = df.copy()

    if 'is_saturated' not in df.columns:
        df['is_saturated'] = (df['P'] <= df['Pb']).astype(int)

    if 'P_over_Pb' not in df.columns:
        df['P_over_Pb'] = df['P'] / df['Pb']

    return df


def group_train_test_split(df, test_size=0.20, seed=42):
    gss = GroupShuffleSplit(n_splits=1, test_size=test_size, random_state=seed)
    groups = df['sample_id'].values

    train_idx, test_idx = next(gss.split(df, groups=groups))

    train_df = df.iloc[train_idx].reset_index(drop=True)
    test_df = df.iloc[test_idx].reset_index(drop=True)

    train_samples = set(train_df['sample_id'].unique())
    test_samples = set(test_df['sample_id'].unique())
    assert len(train_samples & test_samples) == 0, "Data leakage detected!"

    print(f"[Stage 2] Train/test split:")
    print(f"  Train: {len(train_df):,} rows ({len(train_samples):,} samples)")
    print(f"  Test:  {len(test_df):,} rows ({len(test_samples):,} samples)")
    print(f"  Test fraction: {len(test_samples) / (len(train_samples) + len(test_samples)):.2%}")

    return train_df, test_df


def fit_scaler(train_df, feature_cols, save_path=None):
    scaler = StandardScaler()
    scaler.fit(train_df[feature_cols])

    if save_path is not None:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        joblib.dump(scaler, save_path)
        print(f"[Stage 2] Saved scaler to {save_path}")

    return scaler


def run_feature_engineering(input_path='data/synthetic_pvt_data.csv',
                            output_dir='data',
                            models_dir='models',
                            test_size=0.20,
                            seed=42):
    print(f"\n{'='*60}")
    print(f"[Stage 2] Feature Engineering")
    print(f"{'='*60}")

    df = pd.read_csv(input_path)
    print(f"[Stage 2] Loaded {len(df):,} rows from {input_path}")

    df = add_engineered_features(df)

    train_df, test_df = group_train_test_split(df, test_size=test_size, seed=seed)

    feature_cols = ['P', 'API', 'gamma_g', 'gamma_o', 'T', 'Rsb',
                    'Pb', 'P_over_Pb']

    scaler_path = os.path.join(models_dir, 'feature_scaler.pkl')
    scaler = fit_scaler(train_df, feature_cols, save_path=scaler_path)

    os.makedirs(output_dir, exist_ok=True)
    train_path = os.path.join(output_dir, 'train.csv')
    test_path = os.path.join(output_dir, 'test.csv')
    train_df.to_csv(train_path, index=False)
    test_df.to_csv(test_path, index=False)
    print(f"[Stage 2] Saved train set to {train_path}")
    print(f"[Stage 2] Saved test set to {test_path}")

    print(f"\n[Stage 2] Basin distribution:")
    for split_name, split_df in [('Train', train_df), ('Test', test_df)]:
        dist = split_df.groupby('basin_type')['sample_id'].nunique()
        print(f"  {split_name}: {dist.to_dict()}")

    return train_df, test_df, scaler


if __name__ == '__main__':
    run_feature_engineering()
