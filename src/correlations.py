import numpy as np


def api_to_specific_gravity(api):
    return 141.5 / (api + 131.5)


def corrected_gas_gravity_vb(gamma_g, api, Tsep, Psep):
    return gamma_g * (1.0 + 5.912e-5 * api * Tsep * np.log10(Psep / 114.7))


def standing_pb(Rs, gamma_g, api, T):
    A = (Rs / gamma_g) ** 0.83 * 10.0 ** (0.00091 * T - 0.0125 * api)
    Pb = 18.2 * (A - 1.4)
    return np.maximum(Pb, 14.7)

def standing_rs(P, gamma_g, api, T):
    inner = (P / 18.2 + 1.4) * 10.0 ** (0.0125 * api - 0.00091 * T)
    inner = np.maximum(inner, 1e-10)
    Rs = gamma_g * inner ** (1.0 / 0.83)
    return np.maximum(Rs, 0.0)


def standing_bo(Rs, gamma_g, gamma_o, T):
    F = Rs * (gamma_g / gamma_o) ** 0.5 + 1.25 * T
    Bo = 0.9759 + 12.0e-5 * F ** 1.2
    return np.maximum(Bo, 1.0)


def petrosky_pb(Rs, gamma_g, api, T):
    x = 7.916e-4 * np.power(api, 1.5410) - 4.561e-5 * np.power(T, 1.3911)
    Pb = 112.727 * np.power(Rs, 0.5774) / (np.power(gamma_g, 0.8439) * 10.0 ** x) - 12.340
    return np.maximum(Pb, 14.7)


def petrosky_rs(P, gamma_g, api, T):
    x = 7.916e-4 * np.power(api, 1.5410) - 4.561e-5 * np.power(T, 1.3911)
    inner = (P + 12.340) * np.power(gamma_g, 0.8439) * 10.0 ** x / 112.727
    inner = np.maximum(inner, 1e-10)
    Rs = np.power(inner, 1.0 / 0.5774)
    return np.maximum(Rs, 0.0)


def petrosky_bo(Rs, gamma_g, gamma_o, T):
    F = (np.power(Rs, 0.3738)
         * (np.power(gamma_g, 0.2914) / np.power(gamma_o, 0.6265))
         + 0.24626 * np.power(T, 0.5371))
    Bo = 1.0113 + 7.2046e-5 * np.power(F, 3.0936)
    return np.maximum(Bo, 1.0)


def glaso_pb(Rs, gamma_g, api, T):
    A_star = np.power(Rs / gamma_g, 0.816) * np.power(T, 0.172) / np.power(api, 0.989)
    A_star = np.maximum(A_star, 1e-10)
    log_A = np.log10(A_star)
    log_Pb = 1.7669 + 1.7447 * log_A - 0.30218 * log_A ** 2
    Pb = 10.0 ** log_Pb
    return np.maximum(Pb, 14.7)


def glaso_rs(P, gamma_g, api, T):
    P = np.maximum(P, 14.7)
    z = np.log10(P)

    a_coef = 0.30218
    b_coef = -1.7447
    c_coef = z - 1.7669

    discriminant = b_coef ** 2 - 4.0 * a_coef * c_coef
    discriminant = np.maximum(discriminant, 0.0)

    y = (-b_coef - np.sqrt(discriminant)) / (2.0 * a_coef)

    A_star = 10.0 ** y
    Rs = gamma_g * np.power(A_star * np.power(api, 0.989) / np.power(T, 0.172), 1.0 / 0.816)
    return np.maximum(Rs, 0.0)


def glaso_bo(Rs, gamma_g, gamma_o, T):
    Bob_star = Rs * (gamma_g / gamma_o) ** 0.526 + 0.968 * T
    Bob_star = np.maximum(Bob_star, 1.0)
    log_Bob = np.log10(Bob_star)
    log_Bo_minus_1 = -6.58511 + 2.91329 * log_Bob - 0.27683 * log_Bob ** 2
    Bo = 1.0 + 10.0 ** log_Bo_minus_1
    return np.maximum(Bo, 1.0)


def _vb_coefficients(api):
    api = np.asarray(api, dtype=float)
    is_light = api > 30.0

    C1_rs = np.where(is_light, 0.0178, 0.0362)
    C2_rs = np.where(is_light, 1.187, 1.0937)
    C3_rs = np.where(is_light, 23.931, 25.724)

    C1_bo = np.where(is_light, 4.670e-4, 4.677e-4)
    C2_bo = np.where(is_light, 1.100e-5, 1.751e-5)
    C3_bo = np.where(is_light, 1.337e-9, -1.811e-8)

    return C1_rs, C2_rs, C3_rs, C1_bo, C2_bo, C3_bo


def vasquez_beggs_pb(Rs, gamma_g, api, T, Tsep=60.0, Psep=114.7):
    gamma_gc = corrected_gas_gravity_vb(gamma_g, api, Tsep, Psep)
    C1, C2, C3, _, _, _ = _vb_coefficients(api)

    denominator = C1 * gamma_gc * np.exp(C3 * api / (T + 460.0))
    denominator = np.maximum(denominator, 1e-20)
    Pb = np.power(Rs / denominator, 1.0 / C2)
    return np.maximum(Pb, 14.7)


def vasquez_beggs_rs(P, gamma_g, api, T, Tsep=60.0, Psep=114.7):
    gamma_gc = corrected_gas_gravity_vb(gamma_g, api, Tsep, Psep)
    C1, C2, C3, _, _, _ = _vb_coefficients(api)

    Rs = C1 * gamma_gc * np.power(P, C2) * np.exp(C3 * api / (T + 460.0))
    return np.maximum(Rs, 0.0)


def vasquez_beggs_bo(Rs, gamma_g, gamma_o, api, T, Tsep=60.0, Psep=114.7):
    gamma_gc = corrected_gas_gravity_vb(gamma_g, api, Tsep, Psep)
    _, _, _, C1, C2, C3 = _vb_coefficients(api)

    Bo = 1.0 + C1 * Rs + (T - 60.0) * (api / gamma_gc) * (C2 + C3 * Rs)
    return np.maximum(Bo, 1.0)


def beggs_robinson_dead_oil(api, T):
    x = 10.0 ** (3.0324 - 0.02023 * api)
    mu_od = 10.0 ** (x * np.power(T, -1.163)) - 1.0
    return np.maximum(mu_od, 0.01)


def beggs_robinson_saturated(mu_od, Rs):
    a = 10.715 * np.power(Rs + 100.0, -0.515)
    b = 5.44 * np.power(Rs + 150.0, -0.338)
    mu_ob = a * np.power(mu_od, b)
    return np.maximum(mu_ob, 0.01)


def vasquez_beggs_undersaturated_viscosity(mu_ob, P, Pb):
    m = 2.6 * np.power(P, 1.187) * np.exp(-11.513 - 8.98e-5 * P)
    mu_o = mu_ob * np.power(P / Pb, m)
    return np.maximum(mu_o, 0.01)


def vasquez_beggs_oil_compressibility(Rsb, T, gamma_g, api, P, Tsep=60.0, Psep=114.7):
    gamma_gc = corrected_gas_gravity_vb(gamma_g, api, Tsep, Psep)
    numerator = -1433.0 + 5.0 * Rsb + 17.2 * T - 1180.0 * gamma_gc + 12.61 * api
    co = numerator / (1.0e5 * P)
    return np.maximum(co, 1e-7)


def bo_above_bubble_point(Bob, co, P, Pb):
    Bo = Bob * np.exp(co * (Pb - P))
    return np.maximum(Bo, 1.0)


def compute_pb(basin, Rsb, gamma_g, api, T, Tsep=60.0, Psep=114.7):
    dispatch = {
        'A': lambda: standing_pb(Rsb, gamma_g, api, T),
        'B': lambda: petrosky_pb(Rsb, gamma_g, api, T),
        'C': lambda: glaso_pb(Rsb, gamma_g, api, T),
        'D': lambda: vasquez_beggs_pb(Rsb, gamma_g, api, T, Tsep, Psep),
    }
    return dispatch[basin]()


def compute_rs(basin, P, gamma_g, api, T, Tsep=60.0, Psep=114.7):
    dispatch = {
        'A': lambda: standing_rs(P, gamma_g, api, T),
        'B': lambda: petrosky_rs(P, gamma_g, api, T),
        'C': lambda: glaso_rs(P, gamma_g, api, T),
        'D': lambda: vasquez_beggs_rs(P, gamma_g, api, T, Tsep, Psep),
    }
    return dispatch[basin]()


def compute_bo_saturated(basin, Rs, gamma_g, gamma_o, api, T, Tsep=60.0, Psep=114.7):
    dispatch = {
        'A': lambda: standing_bo(Rs, gamma_g, gamma_o, T),
        'B': lambda: petrosky_bo(Rs, gamma_g, gamma_o, T),
        'C': lambda: glaso_bo(Rs, gamma_g, gamma_o, T),
        'D': lambda: vasquez_beggs_bo(Rs, gamma_g, gamma_o, api, T, Tsep, Psep),
    }
    return dispatch[basin]()


def compute_properties_at_pressure(
    basin, P, Pb, Rsb, gamma_g, gamma_o, api, T,
    Tsep=60.0, Psep=114.7
):
    if P <= Pb:
        Rs = compute_rs(basin, P, gamma_g, api, T, Tsep, Psep)
        Rs = min(Rs, Rsb)

        Bo = compute_bo_saturated(basin, Rs, gamma_g, gamma_o, api, T, Tsep, Psep)

        mu_od = beggs_robinson_dead_oil(api, T)
        mu_o = beggs_robinson_saturated(mu_od, Rs)

        is_saturated = 1
    else:
        Rs = Rsb

        Bob = compute_bo_saturated(basin, Rsb, gamma_g, gamma_o, api, T, Tsep, Psep)

        co = vasquez_beggs_oil_compressibility(Rsb, T, gamma_g, api, P, Tsep, Psep)

        Bo = bo_above_bubble_point(Bob, co, P, Pb)

        mu_od = beggs_robinson_dead_oil(api, T)
        mu_ob = beggs_robinson_saturated(mu_od, Rsb)
        mu_o = vasquez_beggs_undersaturated_viscosity(mu_ob, P, Pb)

        is_saturated = 0

    return {
        'Rs': float(Rs),
        'Bo': float(Bo),
        'mu_o': float(mu_o),
        'is_saturated': is_saturated,
    }


def run_correlation_on_dataset(df, correlation_family):
    P = df['P'].values
    api = df['API'].values
    gamma_g = df['gamma_g'].values
    gamma_o = df['gamma_o'].values
    T = df['T'].values
    Rsb = df['Rsb'].values
    Tsep = df['Tsep'].values if 'Tsep' in df.columns else np.full(len(df), 60.0)
    Psep = df['Psep'].values if 'Psep' in df.columns else np.full(len(df), 114.7)
    is_sat = df['is_saturated'].values.astype(bool)

    if correlation_family == 'standing':
        Pb_pred = standing_pb(Rsb, gamma_g, api, T)
    elif correlation_family == 'petrosky':
        Pb_pred = petrosky_pb(Rsb, gamma_g, api, T)
    elif correlation_family == 'glaso':
        Pb_pred = glaso_pb(Rsb, gamma_g, api, T)
    elif correlation_family == 'vasquez_beggs':
        Pb_pred = vasquez_beggs_pb(Rsb, gamma_g, api, T, Tsep, Psep)
    else:
        raise ValueError(f"Unknown correlation family: {correlation_family}")

    Rs_pred = np.zeros(len(df))
    if correlation_family == 'standing':
        Rs_pred[is_sat] = standing_rs(P[is_sat], gamma_g[is_sat], api[is_sat], T[is_sat])
    elif correlation_family == 'petrosky':
        Rs_pred[is_sat] = petrosky_rs(P[is_sat], gamma_g[is_sat], api[is_sat], T[is_sat])
    elif correlation_family == 'glaso':
        Rs_pred[is_sat] = glaso_rs(P[is_sat], gamma_g[is_sat], api[is_sat], T[is_sat])
    elif correlation_family == 'vasquez_beggs':
        Rs_pred[is_sat] = vasquez_beggs_rs(
            P[is_sat], gamma_g[is_sat], api[is_sat], T[is_sat],
            Tsep[is_sat], Psep[is_sat]
        )
    Rs_pred[~is_sat] = Rsb[~is_sat]

    Bo_pred = np.zeros(len(df))
    if correlation_family == 'standing':
        Bo_pred[is_sat] = standing_bo(
            Rs_pred[is_sat], gamma_g[is_sat], gamma_o[is_sat], T[is_sat])
        Bob = standing_bo(Rsb[~is_sat], gamma_g[~is_sat], gamma_o[~is_sat], T[~is_sat])
    elif correlation_family == 'petrosky':
        Bo_pred[is_sat] = petrosky_bo(
            Rs_pred[is_sat], gamma_g[is_sat], gamma_o[is_sat], T[is_sat])
        Bob = petrosky_bo(Rsb[~is_sat], gamma_g[~is_sat], gamma_o[~is_sat], T[~is_sat])
    elif correlation_family == 'glaso':
        Bo_pred[is_sat] = glaso_bo(
            Rs_pred[is_sat], gamma_g[is_sat], gamma_o[is_sat], T[is_sat])
        Bob = glaso_bo(Rsb[~is_sat], gamma_g[~is_sat], gamma_o[~is_sat], T[~is_sat])
    elif correlation_family == 'vasquez_beggs':
        Bo_pred[is_sat] = vasquez_beggs_bo(
            Rs_pred[is_sat], gamma_g[is_sat], gamma_o[is_sat],
            api[is_sat], T[is_sat], Tsep[is_sat], Psep[is_sat])
        Bob = vasquez_beggs_bo(
            Rsb[~is_sat], gamma_g[~is_sat], gamma_o[~is_sat],
            api[~is_sat], T[~is_sat], Tsep[~is_sat], Psep[~is_sat])

    if np.any(~is_sat):
        co = vasquez_beggs_oil_compressibility(
            Rsb[~is_sat], T[~is_sat], gamma_g[~is_sat],
            api[~is_sat], P[~is_sat], Tsep[~is_sat], Psep[~is_sat])
        Bo_pred[~is_sat] = bo_above_bubble_point(Bob, co, P[~is_sat], Pb_pred[~is_sat])

    mu_od = beggs_robinson_dead_oil(api, T)
    mu_o_pred = np.zeros(len(df))
    mu_o_pred[is_sat] = beggs_robinson_saturated(mu_od[is_sat], Rs_pred[is_sat])
    if np.any(~is_sat):
        mu_ob = beggs_robinson_saturated(mu_od[~is_sat], Rsb[~is_sat])
        mu_o_pred[~is_sat] = vasquez_beggs_undersaturated_viscosity(
            mu_ob, P[~is_sat], Pb_pred[~is_sat])

    return {
        'Pb_pred': Pb_pred,
        'Rs_pred': Rs_pred,
        'Bo_pred': Bo_pred,
        'mu_o_pred': mu_o_pred,
    }
