# Eq.py
import numpy as np

# =========================
# Helpers numéricos
# =========================
_EPS = 1e-300

def _as_1d_float(y):
    """Garante saída 1D float para o curve_fit."""
    y = np.asarray(y, dtype=float)
    if y.ndim > 1:
        y = np.squeeze(y)
    return y.ravel().astype(float)

def _safe_log(a):
    return np.log(np.clip(a, _EPS, None))

def _safe_div(num, den):
    den = np.asarray(den, dtype=float)
    return np.divide(num, den, out=np.full_like(den, np.nan, dtype=float), where=den!=0)

def _to_Vars_generic(X, n_vars):
    """
    Converte X em tupla de n_vars vetores (cada um shape (n_obs,)).
    Aceita X: (n_obs, n_vars), (n_vars, n_obs) ou 1D com múltiplo de n_vars.
    """
    X = np.asarray(X, dtype=float)
    if X.ndim == 1:
        if X.size % n_vars != 0:
            raise ValueError(f"Esperado múltiplo de {n_vars} valores em X; recebi {X.size}.")
        X = X.reshape(-1, n_vars)
    elif X.ndim == 2:
        if X.shape[0] == n_vars and X.shape[1] != n_vars:
            X = X.T
        if X.shape[1] != n_vars:
            raise ValueError(f"X deve ter {n_vars} colunas; shape={X.shape}")
    else:
        raise ValueError(f"X deve ter 1 ou 2 dimensões; shape={X.shape}")
    return tuple(X[:, i] for i in range(n_vars))

# =========================
# Funções de projeção (multivariadas)
# =========================
def DMAXPROJETADO_EQ1(Vars, b0):
    IDADE1, IDADE2, DMAX1 = Vars
    frac = _safe_div(IDADE1, IDADE2)
    y = DMAX1 * frac + b0 * (1.0 - frac)
    return _as_1d_float(y)

def DMAXPROJETADO_EQ2(Vars, b0, b1):
    IDADE1, IDADE2, DMAX1 = Vars
    expo = (np.asarray(IDADE2, float) ** b1) - (np.asarray(IDADE1, float) ** b1)
    y = DMAX1 * np.exp(-(b0 ** expo))
    return _as_1d_float(y)

def DMINPROJETADO_EQ3(Vars, b0, b1):
    IDADE1, IDADE2, DMIN1 = Vars
    expo = (np.asarray(IDADE2, float) ** b1) - (np.asarray(IDADE1, float) ** b1)
    y = DMIN1 * np.exp(-(b0 ** expo))
    return _as_1d_float(y)

def BETAPROJETADO_EQ4(Vars, b0):
    IDADE1, IDADE2, DMAX2, BETA1 = Vars
    frac = _safe_div(IDADE1, IDADE2)
    y = BETA1 * frac + b0 * (1.0 - frac) * DMAX2
    return _as_1d_float(y)

def ESCALAPROJETADO_EQ5(Vars, b0):
    IDADE1, IDADE2, ESCALA1 = Vars
    frac = _safe_div(IDADE1, IDADE2)
    y = ESCALA1 * frac + b0 * (1.0 - frac)
    return _as_1d_float(y)

def ESCALAPROJETADO_EQ6(Vars, b0, b1):
    """
    y = exp( ln(ESCALA1) * exp( -b0 * (IDADE2**b1 - IDADE1**b1) ) )
      = ESCALA1 ** exp( -b0 * (IDADE2**b1 - IDADE1**b1) )
    """
    IDADE1, IDADE2, ESCALA1 = Vars
    expo = (np.asarray(IDADE2, float) ** b1) - (np.asarray(IDADE1, float) ** b1)
    u = np.exp(-b0 * expo)
    y = np.exp(_safe_log(ESCALA1) * u)
    return _as_1d_float(y)

def ESCALAPROJETADO_EQ7(Vars, b0, b1, b2):
    """
    y = exp( (ln(ESCALA1)**b0) * exp( -b1 * (IDADE2**b2 - IDADE1**b2) ) )
    """
    IDADE1, IDADE2, ESCALA1 = Vars
    expo = (np.asarray(IDADE2, float) ** b2) - (np.asarray(IDADE1, float) ** b2)
    u = np.exp(-b1 * expo)
    y = np.exp((_safe_log(ESCALA1) ** b0) * u)
    return _as_1d_float(y)

def GOMPERTZ_EQ8(Vars, b1, b2):
    IDADE1, IDADE2, D = Vars
    y = D * np.exp(np.exp(b1 - b2 * IDADE1) - np.exp(b1 - b2 * IDADE2))
    return _as_1d_float(y)

def MODIFIED_EXP_EQ9(Vars, b0):
    IDADE1, IDADE2, D = Vars
    term = _safe_div(1.0, IDADE2) - _safe_div(1.0, IDADE1)
    y = D * np.exp(b0 * term)
    return _as_1d_float(y)

# =========================
# Funções de predição (univariadas)
# =========================
def MODIFIED_EXP_EQ10(x, b0, b1):
    x = np.asarray(x, dtype=float)
    y = b0 * np.exp(_safe_div(b1, x))
    return _as_1d_float(y)

def GOMPERTZ_EQ11(x, b0, b1, b2):
    x = np.asarray(x, dtype=float)
    y = b0 * np.exp(-np.exp(b1 - b2 * x))
    return _as_1d_float(y)

def LOGISTIC_EQ12(x, b0, b1, b2):
    x = np.asarray(x, dtype=float)
    y = _safe_div(b0, (1.0 + b1 * np.exp(-b2 * x)))
    return _as_1d_float(y)

def MMF_EQ13(x, b0, b1, b2, b3):
    x = np.asarray(x, dtype=float)
    y = _safe_div((b0 * b1 + b2 * x ** b3), (b1 + x ** b3))
    return _as_1d_float(y)

def WEIBULL_EQ14(x, b0, b1, b2, b3):
    x = np.asarray(x, dtype=float)
    y = b0 - b1 * np.exp(-b2 * x ** b3)
    return _as_1d_float(y)

def EXPONENTIAL_EQ15(x, b0, b1):
    x = np.asarray(x, dtype=float)
    y = b0 * np.exp(b1 * x)
    return _as_1d_float(y)

# =========================
# Wrappers f(X, *params) para curve_fit
# =========================
def DMAXPROJETADO_EQ1_fit(X, b0):
    Vars = _to_Vars_generic(X, 3)  # [IDADE1, IDADE2, DMAX1]
    return DMAXPROJETADO_EQ1(Vars, b0)

def DMAXPROJETADO_EQ2_fit(X, b0, b1):
    Vars = _to_Vars_generic(X, 3)
    return DMAXPROJETADO_EQ2(Vars, b0, b1)

def DMINPROJETADO_EQ3_fit(X, b0, b1):
    Vars = _to_Vars_generic(X, 3)
    return DMINPROJETADO_EQ3(Vars, b0, b1)

def BETAPROJETADO_EQ4_fit(X, b0):
    Vars = _to_Vars_generic(X, 4)  # [IDADE1, IDADE2, DMAX2, BETA1]
    return BETAPROJETADO_EQ4(Vars, b0)

def ESCALAPROJETADO_EQ5_fit(X, b0):
    Vars = _to_Vars_generic(X, 3)
    return ESCALAPROJETADO_EQ5(Vars, b0)

def ESCALAPROJETADO_EQ6_fit(X, b0, b1):
    Vars = _to_Vars_generic(X, 3)
    return ESCALAPROJETADO_EQ6(Vars, b0, b1)

def ESCALAPROJETADO_EQ7_fit(X, b0, b1, b2):
    Vars = _to_Vars_generic(X, 3)
    return ESCALAPROJETADO_EQ7(Vars, b0, b1, b2)

def GOMPERTZ_EQ8_fit(X, b1, b2):
    Vars = _to_Vars_generic(X, 3)  # [IDADE1, IDADE2, D]
    return GOMPERTZ_EQ8(Vars, b1, b2)

def MODIFIED_EXP_EQ9_fit(X, b0):
    Vars = _to_Vars_generic(X, 3)  # [IDADE1, IDADE2, D]
    return MODIFIED_EXP_EQ9(Vars, b0)

# Univariadas
def MODIFIED_EXP_EQ10_fit(x, b0, b1):  return MODIFIED_EXP_EQ10(x, b0, b1)
def GOMPERTZ_EQ11_fit(x, b0, b1, b2):  return GOMPERTZ_EQ11(x, b0, b1, b2)
def LOGISTIC_EQ12_fit(x, b0, b1, b2):  return LOGISTIC_EQ12(x, b0, b1, b2)
def MMF_EQ13_fit(x, b0, b1, b2, b3):   return MMF_EQ13(x, b0, b1, b2, b3)
def WEIBULL_EQ14_fit(x, b0, b1, b2, b3): return WEIBULL_EQ14(x, b0, b1, b2, b3)
def EXPONENTIAL_EQ15_fit(x, b0, b1):   return EXPONENTIAL_EQ15(x, b0, b1)

# =========================
# Registry (usado pelo Dash)
# =========================
EQUATIONS = {
    # Projeção (multivariadas)
    "DMAXPROJETADO_EQ1": DMAXPROJETADO_EQ1_fit,
    "DMAXPROJETADO_EQ2": DMAXPROJETADO_EQ2_fit,
    "DMINPROJETADO_EQ3": DMINPROJETADO_EQ3_fit,
    "BETAPROJETADO_EQ4": BETAPROJETADO_EQ4_fit,
    "ESCALAPROJETADO_EQ5": ESCALAPROJETADO_EQ5_fit,
    "ESCALAPROJETADO_EQ6": ESCALAPROJETADO_EQ6_fit,
    "ESCALAPROJETADO_EQ7": ESCALAPROJETADO_EQ7_fit,
    "GOMPERTZ_EQ8": GOMPERTZ_EQ8_fit,
    "MODIFIED_EXP_EQ9": MODIFIED_EXP_EQ9_fit,

    # Predição (univariadas)
    "MODIFIED_EXP_EQ10": MODIFIED_EXP_EQ10_fit,
    "GOMPERTZ_EQ11": GOMPERTZ_EQ11_fit,
    "LOGISTIC_EQ12": LOGISTIC_EQ12_fit,
    "MMF_EQ13": MMF_EQ13_fit,
    "WEIBULL_EQ14": WEIBULL_EQ14_fit,
    "EXPONENTIAL_EQ15": EXPONENTIAL_EQ15_fit,
}

# Especificação da ORDEM de variáveis esperada por equação multivariada
# (o app monta X na ordem definida aqui)
VARS_SPEC = {
    "DMAXPROJETADO_EQ1": ["IDADE1", "IDADE2", "DMAX1"],
    "DMAXPROJETADO_EQ2": ["IDADE1", "IDADE2", "DMAX1"],
    "DMINPROJETADO_EQ3": ["IDADE1", "IDADE2", "DMIN1"],
    "BETAPROJETADO_EQ4": ["IDADE1", "IDADE2", "DMAX2", "BETA1"],
    "ESCALAPROJETADO_EQ5": ["IDADE1", "IDADE2", "ESCALA1"],
    "ESCALAPROJETADO_EQ6": ["IDADE1", "IDADE2", "ESCALA1"],
    "ESCALAPROJETADO_EQ7": ["IDADE1", "IDADE2", "ESCALA1"],
    "GOMPERTZ_EQ8": ["IDADE1", "IDADE2", "D"],
    "MODIFIED_EXP_EQ9": ["IDADE1", "IDADE2", "D"],
    # Univariadas
    "MODIFIED_EXP_EQ10": "X",
    "GOMPERTZ_EQ11": "X",
    "LOGISTIC_EQ12": "X",
    "MMF_EQ13": "X",
    "WEIBULL_EQ14": "X",
    "EXPONENTIAL_EQ15": "X",
}
# (Univariadas não entram em VARS_SPEC)

# =========================
# Chutes iniciais (p0)
# =========================
INITIAL_GUESSES = {
    # Multivariadas
    "DMAXPROJETADO_EQ1": [0.5],                # b0
    "DMAXPROJETADO_EQ2": [0.5, 1.0],           # b0, b1
    "DMINPROJETADO_EQ3": [0.5, 1.0],           # b0, b1
    "BETAPROJETADO_EQ4": [0.1],                # b0
    "ESCALAPROJETADO_EQ5": [0.5],              # b0
    "ESCALAPROJETADO_EQ6": [0.1, 1.0],         # b0, b1
    "ESCALAPROJETADO_EQ7": [1.0, 0.1, 1.0],    # b0, b1, b2
    "GOMPERTZ_EQ8": [0.01, 0.1],               # b1, b2
    "MODIFIED_EXP_EQ9": [0.1],                 # b0

    # Univariadas
    "MODIFIED_EXP_EQ10": [1.0, 0.1],           # b0, b1
    "GOMPERTZ_EQ11": [1.0, 0.1, 0.1],          # b0, b1, b2
    "LOGISTIC_EQ12": [1.0, 1.0, 0.1],          # b0, b1, b2
    "MMF_EQ13": [1.0, 1.0, 1.0, 1.0],          # b0, b1, b2, b3
    "WEIBULL_EQ14": [1.0, 1.0, 0.1, 1.0],      # b0, b1, b2, b3
    "EXPONENTIAL_EQ15": [1.0, 0.1],            # b0, b1
}

def initial_guess(name, X, y):
    return list(INITIAL_GUESSES.get(name, []))
