import os
import sys
import time
import datetime

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, PROJECT_ROOT)

from src.data_generation import generate_dataset
from src.feature_engineering import run_feature_engineering
from src.train_models import run_training
from src.validate_physics import run_physics_validation
from src.evaluate import run_evaluation


def generate_report(table_df, physics_results, report_dir='report', plots_dir='plots'):
    print(f"\n{'='*60}")
    print(f"[Stage 7] Report Generation")
    print(f"{'='*60}")

    os.makedirs(report_dir, exist_ok=True)

    ml_row = table_df[table_df['Method'] == 'ML (Best)'].iloc[0]
    best_corr_aape = {}
    targets = ['Pb', 'Rs', 'Bo', 'mu_o']

    for target in targets:
        corr_aapes = table_df[table_df['Method'] != 'ML (Best)'][f'{target}_AAPE%']
        best_corr_aape[target] = corr_aapes.min()

    report = f"""# Basin-Consistent PVT Property Prediction Using Machine Learning

**Date:** {datetime.datetime.now().strftime('%Y-%m-%d')}

## Objective

Develop ML models that predict key PVT properties (Pb, Rs, Bo, mu_o) with accuracy
comparable to or better than classical empirical correlations, using a basin-consistent
training approach that captures geographic variability across 4 correlation families.

## Methodology

### Data Generation
- **{physics_results['n_checked']} synthetic PVT samples** generated via Latin Hypercube Sampling
- Each sample assigned to one of 4 basin types (25% each):
  - **Basin A (Standing):** California-type oils
  - **Basin B (Petrosky-Farshad):** Gulf of Mexico
  - **Basin C (Glaso):** North Sea
  - **Basin D (Vasquez-Beggs):** General/global
- Properties computed self-consistently within each basin's correlation family
- 5% Gaussian noise injected to simulate lab measurement scatter
- ~15 pressure steps per sample (both saturated and undersaturated regions)

### ML Models
| Model | Target | Features | Algorithm |
|-------|--------|----------|-----------|
| 1 | Pb (bubble point) | API, gamma_g, T, Rsb | Best of RF / XGBoost |
| 2 | Rs (solution GOR) | P, T, API, gamma_g, Pb, is_saturated | Best of RF / XGBoost |
| 3 | Bo (oil FVF) | P, T, API, gamma_g, Rs, Pb, is_saturated | Best of RF / XGBoost |
| 4 | mu_o (oil viscosity) | P, T, API, Rs, Pb, is_saturated | Best of RF / XGBoost |

- 5-fold GroupKFold cross-validation (grouped by sample_id to prevent data leakage)
- Hyperparameter tuning over 3 configurations per algorithm

## Key Results

### Performance Comparison (AAPE%)

| Property | Standing | Petrosky-Farshad | Glaso | Vasquez-Beggs | **ML** |
|----------|----------|------------------|-------|---------------|--------|
| Pb | {table_df[table_df['Method']=='Standing'].iloc[0]['Pb_AAPE%']:.2f}% | {table_df[table_df['Method']=='Petrosky-Farshad'].iloc[0]['Pb_AAPE%']:.2f}% | {table_df[table_df['Method']=='Glaso'].iloc[0]['Pb_AAPE%']:.2f}% | {table_df[table_df['Method']=='Vasquez-Beggs'].iloc[0]['Pb_AAPE%']:.2f}% | **{ml_row['Pb_AAPE%']:.2f}%** |
| Rs | {table_df[table_df['Method']=='Standing'].iloc[0]['Rs_AAPE%']:.2f}% | {table_df[table_df['Method']=='Petrosky-Farshad'].iloc[0]['Rs_AAPE%']:.2f}% | {table_df[table_df['Method']=='Glaso'].iloc[0]['Rs_AAPE%']:.2f}% | {table_df[table_df['Method']=='Vasquez-Beggs'].iloc[0]['Rs_AAPE%']:.2f}% | **{ml_row['Rs_AAPE%']:.2f}%** |
| Bo | {table_df[table_df['Method']=='Standing'].iloc[0]['Bo_AAPE%']:.2f}% | {table_df[table_df['Method']=='Petrosky-Farshad'].iloc[0]['Bo_AAPE%']:.2f}% | {table_df[table_df['Method']=='Glaso'].iloc[0]['Bo_AAPE%']:.2f}% | {table_df[table_df['Method']=='Vasquez-Beggs'].iloc[0]['Bo_AAPE%']:.2f}% | **{ml_row['Bo_AAPE%']:.2f}%** |
| mu_o | {table_df[table_df['Method']=='Standing'].iloc[0]['mu_o_AAPE%']:.2f}% | {table_df[table_df['Method']=='Petrosky-Farshad'].iloc[0]['mu_o_AAPE%']:.2f}% | {table_df[table_df['Method']=='Glaso'].iloc[0]['mu_o_AAPE%']:.2f}% | {table_df[table_df['Method']=='Vasquez-Beggs'].iloc[0]['mu_o_AAPE%']:.2f}% | **{ml_row['mu_o_AAPE%']:.2f}%** |

### R2 Scores

| Property | ML R2 |
|----------|-------|
| Pb | {ml_row['Pb_R2']:.5f} |
| Rs | {ml_row['Rs_R2']:.5f} |
| Bo | {ml_row['Bo_R2']:.5f} |
| mu_o | {ml_row['mu_o_R2']:.5f} |

### Ahmed's Published AAE Benchmarks
- Rs: 4.9-12.4%
- Pb: 3.5-7.4%
- Bo: 0.6-2.8%

## Physical Consistency

{physics_results['n_checked']} test samples checked for physical consistency violations:

| Violation Type | Count | Percentage |
|----------------|-------|------------|
| Rs increasing above Pb | {physics_results['violations']['Rs_increases_above_Pb']} | {100*physics_results['violations']['Rs_increases_above_Pb']/max(physics_results['n_checked'],1):.1f}% |
| Bo peak far from Pb | {physics_results['violations']['Bo_peak_far_from_Pb']} | {100*physics_results['violations']['Bo_peak_far_from_Pb']/max(physics_results['n_checked'],1):.1f}% |
| mu_o not decreasing below Pb | {physics_results['violations']['mu_not_decreasing_below_Pb']} | {100*physics_results['violations']['mu_not_decreasing_below_Pb']/max(physics_results['n_checked'],1):.1f}% |

## Limitations

1. Synthetic data only -- trained and tested on correlation-generated data, not real lab measurements
2. No real external validation -- performance on actual field data is unknown
3. Basin assignment is artificial -- real-world oils don't cleanly map to a single correlation family
4. Viscosity uses Beggs-Robinson uniformly -- no basin-specific viscosity calibration
5. Single-well assumption -- each sample is independent; no reservoir continuity modeled

## Future Work

- Validate against real PVT lab reports (Ahmed's 6-oil experimental table)
- Add Marhoun correlation as a 5th basin type (Middle East calibration)
- Implement oil compressibility (co) as an additional ML target
- Explore physics-informed neural networks (PINNs) to enforce monotonicity constraints
- Deploy as a web API for field engineer use
"""

    report_path = os.path.join(report_dir, 'summary.md')
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write(report)

    print(f"[Stage 7] Saved report to {report_path}")
    return report_path


def main():
    start_time = time.time()

    print("=" * 60)
    print("  Basin-Consistent PVT ML Predictor")
    print("  Full Pipeline Execution")
    print("=" * 60)

    data_dir = os.path.join(PROJECT_ROOT, 'data')
    models_dir = os.path.join(PROJECT_ROOT, 'models')
    plots_dir = os.path.join(PROJECT_ROOT, 'plots')
    report_dir = os.path.join(PROJECT_ROOT, 'report')

    csv_path = os.path.join(data_dir, 'synthetic_pvt_data.csv')
    train_path = os.path.join(data_dir, 'train.csv')
    test_path = os.path.join(data_dir, 'test.csv')

    df = generate_dataset(
        n_samples=5000,
        seed=42,
        output_path=csv_path,
    )

    train_df, test_df, scaler = run_feature_engineering(
        input_path=csv_path,
        output_dir=data_dir,
        models_dir=models_dir,
        test_size=0.20,
        seed=42,
    )

    training_results = run_training(
        train_path=train_path,
        test_path=test_path,
        models_dir=models_dir,
        n_cv_folds=5,
    )

    physics_results = run_physics_validation(
        test_path=test_path,
        models_dir=models_dir,
        plots_dir=plots_dir,
        n_plot_samples=6,
        seed=42,
    )

    table_df, corr_preds, ml_preds = run_evaluation(
        test_path=test_path,
        models_dir=models_dir,
        plots_dir=plots_dir,
        report_dir=report_dir,
    )

    report_path = generate_report(table_df, physics_results, report_dir, plots_dir)

    elapsed = time.time() - start_time
    print(f"\n{'='*60}")
    print(f"  SUCCESS Pipeline complete in {elapsed:.1f}s")
    print(f"  Dataset:  {csv_path}")
    print(f"  Models:   {models_dir}/")
    print(f"  Plots:    {plots_dir}/")
    print(f"  Report:   {report_path}")
    print(f"{'='*60}")


if __name__ == '__main__':
    main()
