# PVT Property Prediction Using Machine Learning

## What This Project Does

This project builds ML models to predict four key PVT (Pressure-Volume-Temperature) fluid properties from reservoir inputs. The four targets are bubble-point pressure (Pb), solution gas-oil ratio (Rs), oil formation volume factor (Bo), and oil viscosity (mu_o). Instead of relying on a single empirical correlation, the ML model learns patterns from data generated across four different well-established correlation families, each calibrated to a different geographic basin.

## Why Basin-Consistent Data Generation Matters

A naive approach would be to pick one correlation (say Standing) and train the ML on it. The model would just memorize that one equation and you'd learn nothing. The whole point of this project is to mix data from multiple correlation families so the ML has to generalize across them.

Each sample is assigned to one of four basins:
- Basin A uses Standing (1947) -- originally fit to California crude oils
- Basin B uses Petrosky-Farshad (1993) -- fit to Gulf of Mexico data
- Basin C uses Glaso (1980) -- fit to North Sea oils
- Basin D uses Vasquez-Beggs (1980) -- fit to a global dataset

The assignment is 25% each, shuffled randomly. Within each basin, Pb/Rs/Bo are computed using that basin's own correlation set. Viscosity always uses Beggs-Robinson regardless of basin (there is no basin-specific viscosity correlation in common use).

## Data Generation Details

- 5000 base samples drawn via Latin Hypercube Sampling (LHS) over 6 input dimensions: API gravity (15-45), gas gravity (0.6-1.2), temperature (100-300 F), Rsb (100-2000 scf/STB), separator pressure (50-500 psia), separator temperature (60-120 F)
- LHS is used instead of random sampling because it guarantees better coverage of the input space with fewer samples -- each parameter dimension is evenly stratified
- For each base sample, a pressure sweep of ~15 steps is run from 200 psi to 1.2x Pb, covering both saturated and undersaturated regions
- 5% multiplicative Gaussian noise is added to all output properties to simulate realistic lab measurement scatter
- Invalid samples (Pb out of range, negative Rs, Bo below 1, etc.) are filtered out
- Final dataset is approximately 75,000 rows

## Feature Engineering and Data Splitting

The split is done by sample_id groups using GroupShuffleSplit (80/20 train/test). This is critical -- if you split by random rows, you leak information because multiple pressure points from the same sample share the same Pb, API, T, etc. The model would see part of a sample's pressure curve in training and predict the rest in testing, which inflates metrics.

Two engineered features are added: is_saturated (binary flag for P <= Pb) and P_over_Pb (dimensionless pressure ratio). A StandardScaler is fit on training data and saved, though tree-based models don't strictly need it.

## Model Architecture

Four separate models are trained, one per target:

| Model | Target | Input Features |
|-------|--------|----------------|
| 1 | Pb | API, gamma_g, T, Rsb |
| 2 | Rs | P, T, API, gamma_g, Pb, is_saturated |
| 3 | Bo | P, T, API, gamma_g, Rs, Pb, is_saturated |
| 4 | mu_o | P, T, API, Rs, Pb, is_saturated |

The feature sets are deliberately chained: Pb is a feature for Rs, Bo, and mu_o. Rs is a feature for Bo. This reflects the physical dependency chain.

For each target, both RandomForest and XGBoost are tried with 3 hyperparameter configurations each (6 total candidates). Cross-validation uses 5-fold GroupKFold on sample_id. The best model per target (by CV R2) is selected and refit on the full training set.

## Results

### AAPE% Comparison

| Property | Standing | Petrosky-Farshad | Glaso | Vasquez-Beggs | ML |
|----------|----------|------------------|-------|---------------|-----|
| Pb | 13.51% | 34.04% | 12.84% | 12.86% | 15.91% |
| Rs | 38.83% | 29.09% | 41.41% | 36.28% | 43.58% |
| Bo | 7.38% | 6.74% | 5.97% | 6.34% | 4.81% |
| mu_o | 10.87% | 25.11% | 10.20% | 9.83% | 6.27% |

### R2 Scores (ML)

| Property | R2 |
|----------|----|
| Pb | 0.91052 |
| Rs | 0.91084 |
| Bo | 0.90585 |
| mu_o | 0.99252 |

### Why Pb and Rs AAPE Is High for ML

The ML model was trained on data from 4 different correlation families simultaneously. When you evaluate each individual correlation on its own basin's data, it gets near-zero error on that 25% slice. But AAPE is computed across all basins, so each correlation gets punished on the other 75% of data. The ML model, trying to fit all 4 basins at once, ends up with a compromise that doesn't match any single correlation exactly.

For Pb specifically, the correlation families produce very different Pb values for the same inputs. Standing and Glaso might give 3000 psia while Petrosky gives 5000 psia for the same Rsb/gamma_g/API/T. The ML model averages across these, producing moderate error everywhere rather than zero error on 25% and large error on 75%.

For Rs, the problem is worse because Rs depends on pressure through different functional forms per basin. At low Rs values (near atmospheric pressure), even small absolute errors produce huge percentage errors because the denominator in AAPE is small.

### Why Viscosity R2 Is So High

Viscosity uses Beggs-Robinson uniformly across all basins. There is no basin-splitting for mu_o. The ML model only has to learn one functional relationship (Beggs-Robinson with noise), which is straightforward for tree-based models. This is basically a controlled experiment showing that when there is no multi-correlation ambiguity, the ML model performs extremely well.

### Ahmed's Published AAE Benchmarks (for reference)

These are from Tarek Ahmed's "Reservoir Engineering Handbook" and represent typical correlation accuracy on real lab data:
- Pb: 3.5 to 7.4%
- Rs: 4.9 to 12.4%
- Bo: 0.6 to 2.8%

Our numbers are higher because we're benchmarking against synthetic multi-basin data, not single-correlation real data. The comparison is informational, not apples-to-apples.

## Physical Consistency Validation

Approximately 1000 test samples were checked for three types of physical violations:

| Violation | Count | Rate |
|-----------|-------|------|
| Rs increasing above Pb | 986 | 98.7% |
| Bo peak far from Pb | 162 | 16.2% |
| mu_o not decreasing below Pb | 0 | 0.0% |

The high Rs violation rate is expected -- the Rs model doesn't have a hard constraint that Rs should be flat above Pb. It sees is_saturated=0 as a feature, but nothing forces the prediction to equal Rsb. In production, you'd clamp Rs = Rsb for P > Pb as a post-processing step.

Viscosity has zero violations because the monotonic decrease with pressure (as Rs increases below Pb) is a smooth, consistent pattern that XGBoost captures well.

## Correlations Implemented

All equations follow Ahmed's "Reservoir Engineering Handbook" formulations:

- **Standing (1947):** Pb, Rs, Bo -- originally calibrated to California crude oil data
- **Petrosky-Farshad (1993):** Pb, Rs, Bo -- calibrated to 81 Gulf of Mexico laboratory datasets
- **Glaso (1980):** Pb, Rs, Bo -- calibrated to North Sea oil samples
- **Vasquez-Beggs (1980):** Pb, Rs, Bo -- calibrated to a global dataset of over 5000 measurements, uses corrected gas gravity and API-dependent coefficient switching at API=30
- **Beggs-Robinson (1975):** Dead-oil and saturated viscosity, used across all basins
- **Vasquez-Beggs undersaturated corrections:** Oil compressibility and undersaturated viscosity adjustments for P > Pb

## Pipeline Stages

1. **Data Generation** -- LHS sampling, basin assignment, pressure sweeps, noise injection
2. **Feature Engineering** -- Engineered columns (is_saturated, P_over_Pb), group-aware train/test split
3. **Baseline Correlations** -- Run all 4 correlation families on test set for benchmarking
4. **Model Training** -- RF and XGBoost per target, GroupKFold CV, best model selection
5. **Physical Consistency** -- Check monotonicity constraints on ML predictions
6. **Evaluation** -- RMSE, MAE, R2, AAPE comparison table + residual/parity plots
7. **Report Generation** -- Auto-generated summary with results

## Project Structure

```
PVT_PREDICTOR/
    main.py                     -- orchestrator, runs all 7 stages
    requirements.txt            -- numpy, pandas, scikit-learn, xgboost, matplotlib, seaborn, scipy
    src/
        correlations.py         -- all 4 correlation families + Beggs-Robinson viscosity
        data_generation.py      -- LHS sampling, basin assignment, pressure sweeps
        feature_engineering.py  -- engineered features, group-aware split
        train_models.py         -- RF/XGBoost training with GroupKFold CV
        evaluate.py             -- baseline comparison, metrics, residual/parity plots
        validate_physics.py     -- physical consistency checks and diagnostic plots
    data/
        synthetic_pvt_data.csv  -- full generated dataset (~75K rows)
        train.csv               -- training split
        test.csv                -- test split
    models/
        Pb_model.pkl            -- best Pb model (XGBoost)
        Rs_model.pkl            -- best Rs model
        Bo_model.pkl            -- best Bo model
        mu_o_model.pkl          -- best mu_o model
        feature_scaler.pkl      -- fitted StandardScaler
    plots/
        parity_plots.png        -- predicted vs actual for all 4 properties
        residuals_*.png         -- error vs P and error vs API
        physics_sample_*.png    -- Bo/Rs/mu_o vs P diagnostic curves
    report/
        summary.md              -- this file
        comparison_table.csv    -- full metrics table
```

## Limitations

1. Trained entirely on synthetic correlation-generated data, not real PVT lab reports
2. No external validation on actual field measurements
3. Basin assignment is artificial -- real oils don't map cleanly to one correlation
4. Viscosity has no basin-specific calibration (Beggs-Robinson everywhere)
5. Each sample is treated independently; no spatial or reservoir continuity is modeled
6. The Rs model doesn't enforce Rs = Rsb above Pb (needs post-processing clamp)

## Possible Extensions

- Validate on real PVT lab data from Ahmed's 6-oil experimental dataset
- Add Marhoun (1988) as a 5th basin type for Middle East calibration
- Train a separate oil compressibility model
- Use physics-informed neural networks (PINNs) to enforce monotonicity as a loss constraint rather than a post-processing check
- Deploy as a REST API for field engineers
