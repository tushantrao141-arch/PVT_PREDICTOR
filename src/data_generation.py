import os
import numpy as np
import pandas as pd
from scipy.stats import qmc

from src.correlations import (
    api_to_specific_gravity,
    compute_pb,
    compute_properties_at_pressure,
)


PARAM_RANGES = {
    'API':   (15.0, 45.0),
    'gamma_g': (0.60, 1.20),
    'T':     (100.0, 300.0),
    'Rsb':   (100.0, 2000.0),
    'Psep':  (50.0, 500.0),
    'Tsep':  (60.0, 120.0),
}

BASIN_TYPES = ['A', 'B', 'C', 'D']
N_PRESSURE_STEPS = 15
NOISE_STD = 0.05


def generate_lhs_samples(n_samples, seed=42):
    param_names = list(PARAM_RANGES.keys())
    n_params = len(param_names)

    sampler = qmc.LatinHypercube(d=n_params, seed=seed)
    raw = sampler.random(n=n_samples)
    lower = np.array([PARAM_RANGES[p][0] for p in param_names])
    upper = np.array([PARAM_RANGES[p][1] for p in param_names])
    scaled = qmc.scale(raw, lower, upper)
    df = pd.DataFrame(scaled, columns=param_names)
    return df


def assign_basins(n_samples, seed=42):
    rng = np.random.default_rng(seed)
    basins = np.repeat(BASIN_TYPES, n_samples // len(BASIN_TYPES))

    remainder = n_samples - len(basins)
    if remainder > 0:
        basins = np.concatenate([basins, rng.choice(BASIN_TYPES, remainder)])

    rng.shuffle(basins)
    return basins


def generate_pressure_sweep(Pb, n_steps=N_PRESSURE_STEPS):
    P_min = 200.0
    P_max = max(1.2 * Pb, Pb + 500.0)

    n_below = max(int(0.6 * n_steps), 3)
    n_above = n_steps - n_below

    P_below = np.linspace(P_min, Pb, n_below, endpoint=True)
    P_above = np.linspace(Pb + (P_max - Pb) / n_above, P_max, n_above)

    pressures = np.concatenate([P_below, P_above])
    return np.sort(np.unique(pressures))


def add_gaussian_noise(value, noise_std=NOISE_STD, rng=None):
    if rng is None:
        rng = np.random.default_rng()
    noise_factor = 1.0 + rng.normal(0.0, noise_std)
    return max(value * noise_factor, 0.0)


def generate_dataset(n_samples=5000, seed=42, output_path=None):
    print(f"[Stage 1] Generating {n_samples} base samples with LHS...")

    samples = generate_lhs_samples(n_samples, seed=seed)

    samples['basin_type'] = assign_basins(n_samples, seed=seed)

    samples['gamma_o'] = api_to_specific_gravity(samples['API'].values)

    print("[Stage 1] Computing bubble-point pressures per basin...")
    Pb_values = np.zeros(n_samples)
    for i in range(n_samples):
        row = samples.iloc[i]
        try:
            Pb_values[i] = compute_pb(
                basin=row['basin_type'],
                Rsb=row['Rsb'],
                gamma_g=row['gamma_g'],
                api=row['API'],
                T=row['T'],
                Tsep=row['Tsep'],
                Psep=row['Psep'],
            )
        except Exception:
            Pb_values[i] = np.nan
    samples['Pb'] = Pb_values

    valid_mask = (
        samples['Pb'].notna()
        & (samples['Pb'] > 100)
        & (samples['Pb'] < 15000)
    )
    samples = samples[valid_mask].reset_index(drop=True)
    n_valid = len(samples)
    print(f"[Stage 1] {n_valid}/{n_samples} samples have valid Pb values.")

    print("[Stage 1] Generating pressure sweeps and computing PVT properties")
    rng = np.random.default_rng(seed + 1)

    all_rows = []
    for sample_id in range(n_valid):
        row = samples.iloc[sample_id]
        Pb = row['Pb']
        pressures = generate_pressure_sweep(Pb)

        for P in pressures:
            try:
                props = compute_properties_at_pressure(
                    basin=row['basin_type'],
                    P=P,
                    Pb=Pb,
                    Rsb=row['Rsb'],
                    gamma_g=row['gamma_g'],
                    gamma_o=row['gamma_o'],
                    api=row['API'],
                    T=row['T'],
                    Tsep=row['Tsep'],
                    Psep=row['Psep'],
                )
            except Exception:
                continue

            if (props['Rs'] < 0 or props['Bo'] < 1.0 or props['mu_o'] < 0.01
                    or np.isnan(props['Rs']) or np.isnan(props['Bo'])
                    or np.isnan(props['mu_o'])):
                continue

            Rs_noisy = add_gaussian_noise(props['Rs'], NOISE_STD, rng)
            Bo_noisy = max(add_gaussian_noise(props['Bo'], NOISE_STD, rng), 1.0)
            mu_o_noisy = max(add_gaussian_noise(props['mu_o'], NOISE_STD, rng), 0.01)
            Pb_noisy = add_gaussian_noise(Pb, NOISE_STD, rng)

            all_rows.append({
                'sample_id': sample_id,
                'basin_type': row['basin_type'],
                'API': row['API'],
                'gamma_g': row['gamma_g'],
                'gamma_o': row['gamma_o'],
                'T': row['T'],
                'Rsb': row['Rsb'],
                'Psep': row['Psep'],
                'Tsep': row['Tsep'],
                'P': P,
                'Pb': Pb_noisy,
                'Rs': Rs_noisy,
                'Bo': Bo_noisy,
                'mu_o': mu_o_noisy,
                'is_saturated': props['is_saturated'],
                'P_over_Pb': P / Pb,
            })

        if (sample_id + 1) % 1000 == 0:
            print(f"processed {sample_id + 1}/{n_valid} samples")

    df = pd.DataFrame(all_rows)
    print(f"[Stage 1] Generated {len(df)} total rows from {n_valid} base samples.")

    print(f"\n[Stage 1] Dataset Summary:")
    print(f"  Rows: {len(df):,}")
    print(f"  Samples: {df['sample_id'].nunique():,}")
    print(f"  Basin distribution: {df.groupby('basin_type')['sample_id'].nunique().to_dict()}")
    print(f"  Saturated rows: {df['is_saturated'].sum():,} "
          f"({100 * df['is_saturated'].mean():.1f}%)")
    print(f"  Pb range: {df['Pb'].min():.0f} - {df['Pb'].max():.0f} psia")
    print(f"  Rs range: {df['Rs'].min():.1f} - {df['Rs'].max():.1f} scf/STB")
    print(f"  Bo range: {df['Bo'].min():.3f} - {df['Bo'].max():.3f}")
    print(f"  mu_o range: {df['mu_o'].min():.3f} - {df['mu_o'].max():.3f} cp")

    if output_path is not None:
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        df.to_csv(output_path, index=False)
        print(f"\n[Stage 1] Saved dataset to {output_path}")

    return df


if __name__ == '__main__':
    df = generate_dataset(
        n_samples=5000,
        seed=42,
        output_path='data/synthetic_pvt_data.csv',
    )
