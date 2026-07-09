import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import joblib

from src.correlations import run_correlation_on_dataset
from src.train_models import MODEL_CONFIGS


AHMED_AAE_BENCHMARKS = {
    'Rs': {'min': 4.9, 'max': 12.4, 'label': '4.9-12.4%'},
    'Pb': {'min': 3.5, 'max': 7.4, 'label': '3.5-7.4%'},
    'Bo': {'min': 0.6, 'max': 2.8, 'label': '0.6-2.8%'},
}

CORRELATION_NAMES = ['standing', 'petrosky', 'glaso', 'vasquez_beggs']
CORRELATION_LABELS = {
    'standing': 'Standing',
    'petrosky': 'Petrosky-Farshad',
    'glaso': 'Glaso',
    'vasquez_beggs': 'Vasquez-Beggs',
}


def compute_metrics(y_true, y_pred, label=''):
    mask = np.abs(y_true) > 1e-6
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mae = mean_absolute_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)
    aape = np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100

    return {'label': label, 'RMSE': rmse, 'MAE': mae, 'R2': r2, 'AAPE%': aape}


def run_baseline_correlations(test_df):
    print(f"\n{'='*60}")
    print(f"[Stage 3] Baseline Correlations on Test Set")
    print(f"{'='*60}")

    all_predictions = {}
    for corr_name in CORRELATION_NAMES:
        print(f"  Running {CORRELATION_LABELS[corr_name]}...")
        preds = run_correlation_on_dataset(test_df, corr_name)
        all_predictions[corr_name] = preds

    return all_predictions


def run_ml_predictions(test_df, models_dir='models'):
    predictions = {}

    for model_name, config in MODEL_CONFIGS.items():
        model_path = os.path.join(models_dir, f'{model_name}_model.pkl')
        model = joblib.load(model_path)

        X_test = test_df[config['features']].values
        y_pred = model.predict(X_test)
        predictions[config['target']] = y_pred

    return predictions


def build_comparison_table(test_df, corr_predictions, ml_predictions):
    targets = ['Pb', 'Rs', 'Bo', 'mu_o']
    corr_target_map = {
        'Pb': 'Pb_pred', 'Rs': 'Rs_pred', 'Bo': 'Bo_pred', 'mu_o': 'mu_o_pred'
    }

    rows = []

    for corr_name in CORRELATION_NAMES:
        row = {'Method': CORRELATION_LABELS[corr_name]}
        preds = corr_predictions[corr_name]

        for target in targets:
            y_true = test_df[target].values
            y_pred = preds[corr_target_map[target]]
            metrics = compute_metrics(y_true, y_pred)
            row[f'{target}_RMSE'] = metrics['RMSE']
            row[f'{target}_MAE'] = metrics['MAE']
            row[f'{target}_R2'] = metrics['R2']
            row[f'{target}_AAPE%'] = metrics['AAPE%']

        rows.append(row)

    ml_row = {'Method': 'ML (Best)'}
    for target in targets:
        y_true = test_df[target].values
        y_pred = ml_predictions[target]
        metrics = compute_metrics(y_true, y_pred)
        ml_row[f'{target}_RMSE'] = metrics['RMSE']
        ml_row[f'{target}_MAE'] = metrics['MAE']
        ml_row[f'{target}_R2'] = metrics['R2']
        ml_row[f'{target}_AAPE%'] = metrics['AAPE%']

    rows.append(ml_row)

    return pd.DataFrame(rows)


def print_comparison_table(table_df):
    targets = ['Pb', 'Rs', 'Bo', 'mu_o']

    print(f"\n{'='*80}")
    print(f"[Stage 6] === COMPARISON TABLE: AAPE% ===")
    print(f"{'='*80}")
    header = f"{'Method':<20}"
    for t in targets:
        header += f"{'  ' + t + ' AAPE%':<14}"
    print(header)
    print("-" * 76)

    for _, row in table_df.iterrows():
        line = f"{row['Method']:<20}"
        for t in targets:
            line += f"  {row[f'{t}_AAPE%']:>9.2f}%  "
        print(line)

    print(f"\nAhmed's Published AAE Benchmarks:")
    for prop, bench in AHMED_AAE_BENCHMARKS.items():
        print(f"  {prop}: {bench['label']}")

    print(f"\n{'='*80}")
    print(f"[Stage 6] === COMPARISON TABLE: R2 ===")
    print(f"{'='*80}")
    header = f"{'Method':<20}"
    for t in targets:
        header += f"{'  ' + t + ' R2':<14}"
    print(header)
    print("-" * 76)

    for _, row in table_df.iterrows():
        line = f"{row['Method']:<20}"
        for t in targets:
            line += f"  {row[f'{t}_R2']:>9.5f}  "
        print(line)


def plot_residuals(test_df, corr_predictions, ml_predictions, plots_dir='plots'):
    os.makedirs(plots_dir, exist_ok=True)
    targets = ['Pb', 'Rs', 'Bo', 'mu_o']
    target_labels = {'Pb': 'Pb (psia)', 'Rs': 'Rs (scf/STB)',
                     'Bo': 'Bo (bbl/STB)', 'mu_o': 'mu_o (cp)'}
    corr_target_map = {
        'Pb': 'Pb_pred', 'Rs': 'Rs_pred', 'Bo': 'Bo_pred', 'mu_o': 'mu_o_pred'
    }

    sns.set_theme(style='whitegrid', font_scale=1.1)

    for target in targets:
        y_true = test_df[target].values
        mask = np.abs(y_true) > 1e-6

        fig, axes = plt.subplots(1, 2, figsize=(16, 6))
        fig.suptitle(f'{target_labels[target]} — Residual Analysis', fontsize=14, fontweight='bold')

        for ax_idx, x_var in enumerate(['P', 'API']):
            ax = axes[ax_idx]
            x_data = test_df[x_var].values

            for corr_name in CORRELATION_NAMES:
                y_pred = corr_predictions[corr_name][corr_target_map[target]]
                pct_error = np.where(mask, (y_pred - y_true) / y_true * 100, 0)
                ax.scatter(x_data[mask], pct_error[mask], alpha=0.05, s=3,
                           label=CORRELATION_LABELS[corr_name])

            y_pred_ml = ml_predictions[target]
            pct_error_ml = np.where(mask, (y_pred_ml - y_true) / y_true * 100, 0)
            ax.scatter(x_data[mask], pct_error_ml[mask], alpha=0.15, s=5,
                       color='black', label='ML', zorder=5)

            ax.axhline(0, color='red', linestyle='--', linewidth=0.8)
            ax.set_xlabel(x_var)
            ax.set_ylabel('% Error')
            ax.set_title(f'Error vs {x_var}')
            ax.legend(fontsize=8, markerscale=5)
            ax.set_ylim(-50, 50)

        plt.tight_layout()
        save_path = os.path.join(plots_dir, f'residuals_{target}.png')
        fig.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close(fig)
        print(f"  Saved {save_path}")


def plot_parity(test_df, ml_predictions, plots_dir='plots'):
    os.makedirs(plots_dir, exist_ok=True)
    targets = ['Pb', 'Rs', 'Bo', 'mu_o']
    target_labels = {'Pb': 'Pb (psia)', 'Rs': 'Rs (scf/STB)',
                     'Bo': 'Bo (bbl/STB)', 'mu_o': 'mu_o (cp)'}

    fig, axes = plt.subplots(2, 2, figsize=(14, 14))
    fig.suptitle('ML Model — Parity Plots (Predicted vs Actual)', fontsize=14, fontweight='bold')

    for idx, target in enumerate(targets):
        ax = axes[idx // 2, idx % 2]
        y_true = test_df[target].values
        y_pred = ml_predictions[target]

        r2 = r2_score(y_true, y_pred)

        ax.scatter(y_true, y_pred, alpha=0.1, s=3, color='steelblue')

        lims = [min(y_true.min(), y_pred.min()), max(y_true.max(), y_pred.max())]
        ax.plot(lims, lims, 'r--', linewidth=1, label='Perfect prediction')

        ax.set_xlabel(f'Actual {target_labels[target]}')
        ax.set_ylabel(f'Predicted {target_labels[target]}')
        ax.set_title(f'{target_labels[target]} (R2 = {r2:.5f})')
        ax.legend()

    plt.tight_layout()
    save_path = os.path.join(plots_dir, 'parity_plots.png')
    fig.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"  Saved {save_path}")


def run_evaluation(test_path='data/test.csv',
                   models_dir='models',
                   plots_dir='plots',
                   report_dir='report'):
    print(f"\n{'='*60}")
    print(f"[Stage 6] Evaluation & Comparison")
    print(f"{'='*60}")

    test_df = pd.read_csv(test_path)
    print(f"[Stage 6] Test set: {len(test_df):,} rows")

    corr_predictions = run_baseline_correlations(test_df)

    print(f"\n[Stage 6] Loading ML models and generating predictions...")
    ml_predictions = run_ml_predictions(test_df, models_dir)

    table_df = build_comparison_table(test_df, corr_predictions, ml_predictions)
    print_comparison_table(table_df)

    os.makedirs(report_dir, exist_ok=True)
    table_path = os.path.join(report_dir, 'comparison_table.csv')
    table_df.to_csv(table_path, index=False)
    print(f"\n[Stage 6] Saved comparison table to {table_path}")

    print(f"\n[Stage 6] Generating residual plots...")
    plot_residuals(test_df, corr_predictions, ml_predictions, plots_dir)

    print(f"\n[Stage 6] Generating parity plots...")
    plot_parity(test_df, ml_predictions, plots_dir)

    return table_df, corr_predictions, ml_predictions


if __name__ == '__main__':
    run_evaluation()
