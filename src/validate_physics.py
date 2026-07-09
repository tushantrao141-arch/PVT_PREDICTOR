import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import joblib

from src.train_models import MODEL_CONFIGS


def load_models(models_dir='models'):
    models = {}
    for name in MODEL_CONFIGS:
        path = os.path.join(models_dir, f'{name}_model.pkl')
        models[name] = joblib.load(path)
    return models


def predict_for_sample(sample_df, models):
    result = sample_df.copy()

    for model_name, config in MODEL_CONFIGS.items():
        features = config['features']
        target = config['target']
        model = models[model_name]

        if model_name == 'Pb':
            X = sample_df[features].iloc[:1].values
            pb_pred = model.predict(X)[0]
            result['Pb_pred'] = pb_pred
        else:
            X = sample_df[features].values
            result[f'{target}_pred'] = model.predict(X)

    return result


def check_physical_consistency(sample_df, predictions):
    P = predictions['P'].values
    Pb = predictions['Pb'].values[0]
    Pb_pred = predictions['Pb_pred'].values[0] if 'Pb_pred' in predictions.columns else Pb

    violations = {}

    above_pb = P > Pb
    if np.sum(above_pb) > 1:
        Rs_above = predictions.loc[above_pb, 'Rs_pred'].values
        if len(Rs_above) > 1:
            Rs_increasing = np.any(np.diff(Rs_above) > 0.01 * np.mean(Rs_above))
            violations['Rs_increases_above_Pb'] = bool(Rs_increasing)

    if 'Bo_pred' in predictions.columns:
        Bo_pred = predictions['Bo_pred'].values
        Bo_peak_idx = np.argmax(Bo_pred)
        P_at_peak = P[Bo_peak_idx]
        violations['Bo_peak_far_from_Pb'] = abs(P_at_peak - Pb) / Pb > 0.20

    if 'mu_o_pred' in predictions.columns:
        below_pb = P <= Pb
        above_pb = P > Pb
        if np.sum(below_pb) > 2 and np.sum(above_pb) > 2:
            mu_below = predictions.loc[below_pb, 'mu_o_pred'].values
            mu_trend_below = np.polyfit(P[below_pb], mu_below, 1)[0]
            violations['mu_not_decreasing_below_Pb'] = mu_trend_below > 0

    return violations


def plot_sample_physics(predictions, sample_id, basin_type, plots_dir='plots'):
    P = predictions['P'].values
    Pb = predictions['Pb'].values[0]

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    fig.suptitle(
        f'Sample {sample_id} (Basin {basin_type}) — Physical Consistency Check',
        fontsize=13, fontweight='bold'
    )

    ax = axes[0]
    ax.plot(P, predictions['Rs'].values, 'b-', linewidth=2, label='True (noisy)', alpha=0.7)
    if 'Rs_pred' in predictions.columns:
        ax.plot(P, predictions['Rs_pred'].values, 'r--', linewidth=2, label='ML Predicted')
    ax.axvline(Pb, color='green', linestyle=':', linewidth=1.5, label=f'Pb={Pb:.0f}')
    ax.set_xlabel('Pressure (psia)')
    ax.set_ylabel('Rs (scf/STB)')
    ax.set_title('Solution GOR vs Pressure')
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)

    ax = axes[1]
    ax.plot(P, predictions['Bo'].values, 'b-', linewidth=2, label='True (noisy)', alpha=0.7)
    if 'Bo_pred' in predictions.columns:
        ax.plot(P, predictions['Bo_pred'].values, 'r--', linewidth=2, label='ML Predicted')
    ax.axvline(Pb, color='green', linestyle=':', linewidth=1.5, label=f'Pb={Pb:.0f}')
    ax.set_xlabel('Pressure (psia)')
    ax.set_ylabel('Bo (bbl/STB)')
    ax.set_title('Oil FVF vs Pressure')
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)

    ax = axes[2]
    ax.plot(P, predictions['mu_o'].values, 'b-', linewidth=2, label='True (noisy)', alpha=0.7)
    if 'mu_o_pred' in predictions.columns:
        ax.plot(P, predictions['mu_o_pred'].values, 'r--', linewidth=2, label='ML Predicted')
    ax.axvline(Pb, color='green', linestyle=':', linewidth=1.5, label=f'Pb={Pb:.0f}')
    ax.set_xlabel('Pressure (psia)')
    ax.set_ylabel('mu_o (cp)')
    ax.set_title('Oil Viscosity vs Pressure')
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    save_path = os.path.join(plots_dir, f'physics_sample_{sample_id}.png')
    fig.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    return save_path


def run_physics_validation(test_path='data/test.csv',
                           models_dir='models',
                           plots_dir='plots',
                           n_plot_samples=6,
                           seed=42):
    print(f"\n{'='*60}")
    print(f"[Stage 5] Physical Consistency Validation")
    print(f"{'='*60}")

    test_df = pd.read_csv(test_path)
    models = load_models(models_dir)
    os.makedirs(plots_dir, exist_ok=True)

    rng = np.random.default_rng(seed)
    sample_ids = test_df['sample_id'].unique()

    selected_samples = []
    for basin in ['A', 'B', 'C', 'D']:
        basin_samples = test_df[test_df['basin_type'] == basin]['sample_id'].unique()
        if len(basin_samples) > 0:
            selected_samples.append(rng.choice(basin_samples))

    remaining = n_plot_samples - len(selected_samples)
    if remaining > 0:
        extra = rng.choice(
            [s for s in sample_ids if s not in selected_samples],
            min(remaining, len(sample_ids) - len(selected_samples)),
            replace=False
        )
        selected_samples.extend(extra)

    print(f"[Stage 5] Checking {len(selected_samples)} samples for physical consistency...\n")

    total_violations = {
        'Rs_increases_above_Pb': 0,
        'Bo_peak_far_from_Pb': 0,
        'mu_not_decreasing_below_Pb': 0,
    }
    plot_paths = []

    for sample_id in selected_samples:
        sample_df = test_df[test_df['sample_id'] == sample_id].sort_values('P')
        basin_type = sample_df['basin_type'].values[0]

        predictions = predict_for_sample(sample_df, models)

        violations = check_physical_consistency(sample_df, predictions)

        for key, violated in violations.items():
            if violated:
                total_violations[key] += 1

        status = "PASS" if not any(violations.values()) else "VIOLATIONS"
        print(f"  Sample {sample_id} (Basin {basin_type}): {status}")
        for key, violated in violations.items():
            if violated:
                print(f"    WARNING: {key}")

        path = plot_sample_physics(predictions, sample_id, basin_type, plots_dir)
        plot_paths.append(path)

    print(f"\n[Stage 5] Scanning entire test set for violations...")
    all_sample_ids = test_df['sample_id'].unique()
    full_violations = {k: 0 for k in total_violations}
    n_checked = 0

    for sample_id in all_sample_ids:
        sample_df = test_df[test_df['sample_id'] == sample_id].sort_values('P')
        if len(sample_df) < 5:
            continue

        predictions = predict_for_sample(sample_df, models)
        violations = check_physical_consistency(sample_df, predictions)

        for key, violated in violations.items():
            if violated:
                full_violations[key] += 1
        n_checked += 1

    print(f"\n[Stage 5] === Violation Summary ({n_checked} samples checked) ===")
    for key, count in full_violations.items():
        pct = 100 * count / max(n_checked, 1)
        print(f"  {key}: {count}/{n_checked} ({pct:.1f}%)")

    print(f"\n[Stage 5] Saved {len(plot_paths)} diagnostic plots to {plots_dir}/")

    return {
        'n_checked': n_checked,
        'violations': full_violations,
        'plot_paths': plot_paths,
    }


if __name__ == '__main__':
    run_physics_validation()
