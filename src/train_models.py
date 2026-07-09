import os
import warnings
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import GroupKFold, cross_val_score
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import xgboost as xgb
import joblib

warnings.filterwarnings('ignore', category=UserWarning)


MODEL_CONFIGS = {
    'Pb': {
        'features': ['API', 'gamma_g', 'T', 'Rsb'],
        'target': 'Pb',
        'description': 'Bubble-Point Pressure',
    },
    'Rs': {
        'features': ['P', 'T', 'API', 'gamma_g', 'Pb', 'is_saturated'],
        'target': 'Rs',
        'description': 'Solution Gas-Oil Ratio',
    },
    'Bo': {
        'features': ['P', 'T', 'API', 'gamma_g', 'Rs', 'Pb', 'is_saturated'],
        'target': 'Bo',
        'description': 'Oil Formation Volume Factor',
    },
    'mu_o': {
        'features': ['P', 'T', 'API', 'Rs', 'Pb', 'is_saturated'],
        'target': 'mu_o',
        'description': 'Oil Viscosity',
    },
}


RF_PARAM_GRID = [
    {'n_estimators': 200, 'max_depth': 15, 'min_samples_split': 5},
    {'n_estimators': 300, 'max_depth': 20, 'min_samples_split': 3},
    {'n_estimators': 400, 'max_depth': 25, 'min_samples_split': 2},
]

XGB_PARAM_GRID = [
    {'n_estimators': 200, 'max_depth': 6, 'learning_rate': 0.1, 'subsample': 0.8},
    {'n_estimators': 300, 'max_depth': 8, 'learning_rate': 0.05, 'subsample': 0.8},
    {'n_estimators': 500, 'max_depth': 10, 'learning_rate': 0.03, 'subsample': 0.9},
]


def train_single_model(train_df, model_name, n_cv_folds=5, verbose=True):
    config = MODEL_CONFIGS[model_name]
    features = config['features']
    target = config['target']

    X = train_df[features].values
    y = train_df[target].values
    groups = train_df['sample_id'].values

    if verbose:
        print(f"\n  Training {model_name} ({config['description']})...")
        print(f"    Features: {features}")
        print(f"    Samples: {len(X):,} rows")

    gkf = GroupKFold(n_splits=n_cv_folds)
    best_model = None
    best_score = -np.inf
    best_type = None
    results = []

    for i, params in enumerate(RF_PARAM_GRID):
        rf = RandomForestRegressor(
            n_estimators=params['n_estimators'],
            max_depth=params['max_depth'],
            min_samples_split=params['min_samples_split'],
            random_state=42,
            n_jobs=-1,
        )
        scores = cross_val_score(
            rf, X, y, groups=groups, cv=gkf,
            scoring='r2', n_jobs=-1,
        )
        mean_r2 = scores.mean()
        results.append(('RF', params, mean_r2, scores.std()))

        if mean_r2 > best_score:
            best_score = mean_r2
            best_type = f'RandomForest (config {i+1})'
            rf.fit(X, y)
            best_model = rf

        if verbose:
            print(f"    RF config {i+1}: R2={mean_r2:.5f} +/- {scores.std():.5f}")

    for i, params in enumerate(XGB_PARAM_GRID):
        xgb_model = xgb.XGBRegressor(
            n_estimators=params['n_estimators'],
            max_depth=params['max_depth'],
            learning_rate=params['learning_rate'],
            subsample=params['subsample'],
            random_state=42,
            n_jobs=-1,
            verbosity=0,
        )
        scores = cross_val_score(
            xgb_model, X, y, groups=groups, cv=gkf,
            scoring='r2', n_jobs=-1,
        )
        mean_r2 = scores.mean()
        results.append(('XGB', params, mean_r2, scores.std()))

        if mean_r2 > best_score:
            best_score = mean_r2
            best_type = f'XGBoost (config {i+1})'
            xgb_model.fit(X, y)
            best_model = xgb_model

        if verbose:
            print(f"    XGB config {i+1}: R2={mean_r2:.5f} +/- {scores.std():.5f}")

    if verbose:
        print(f"    SUCCESS Best: {best_type}, CV R2={best_score:.5f}")

    return best_model, best_type, best_score, results


def evaluate_on_test(model, test_df, model_name):
    config = MODEL_CONFIGS[model_name]
    features = config['features']
    target = config['target']

    X_test = test_df[features].values
    y_test = test_df[target].values
    y_pred = model.predict(X_test)

    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    mask = np.abs(y_test) > 1e-6
    aape = np.mean(np.abs((y_test[mask] - y_pred[mask]) / y_test[mask])) * 100

    return {
        'model_name': model_name,
        'rmse': rmse,
        'mae': mae,
        'r2': r2,
        'aape': aape,
        'y_test': y_test,
        'y_pred': y_pred,
    }


def run_training(train_path='data/train.csv',
                 test_path='data/test.csv',
                 models_dir='models',
                 n_cv_folds=5):
    print(f"\n{'='*60}")
    print(f"[Stage 4] Model Training (4 separate models)")
    print(f"{'='*60}")

    train_df = pd.read_csv(train_path)
    test_df = pd.read_csv(test_path)
    print(f"[Stage 4] Train: {len(train_df):,} rows, Test: {len(test_df):,} rows")

    os.makedirs(models_dir, exist_ok=True)
    all_results = {}

    for model_name in MODEL_CONFIGS:
        model, model_type, cv_score, cv_results = train_single_model(
            train_df, model_name, n_cv_folds=n_cv_folds
        )

        test_metrics = evaluate_on_test(model, test_df, model_name)

        model_path = os.path.join(models_dir, f'{model_name}_model.pkl')
        joblib.dump(model, model_path)
        print(f"    Saved to {model_path}")

        print(f"    Test set: RMSE={test_metrics['rmse']:.4f}, "
              f"MAE={test_metrics['mae']:.4f}, R2={test_metrics['r2']:.5f}, "
              f"AAPE={test_metrics['aape']:.2f}%")

        all_results[model_name] = {
            'model': model,
            'model_type': model_type,
            'cv_score': cv_score,
            'test_metrics': test_metrics,
        }

    print(f"\n[Stage 4] === ML Model Performance Summary ===")
    print(f"{'Model':<8} {'Type':<28} {'CV R2':<10} {'Test R2':<10} "
          f"{'Test RMSE':<12} {'Test AAPE%':<10}")
    print("-" * 78)
    for name, res in all_results.items():
        tm = res['test_metrics']
        print(f"{name:<8} {res['model_type']:<28} {res['cv_score']:<10.5f} "
              f"{tm['r2']:<10.5f} {tm['rmse']:<12.4f} {tm['aape']:<10.2f}")

    return all_results


if __name__ == '__main__':
    run_training()
