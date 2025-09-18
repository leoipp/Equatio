# callbacks.py
import base64, io
from uuid import uuid4

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

from dash import Input, Output, State, ctx, no_update, dcc, html
from dash.dependencies import ALL
from scipy.optimize import curve_fit

from .Eq import compile_sympy_univar, compile_sympy_vars

from flask_login import current_user
from ..models import db, UserModel

import logging
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

# === Registries ===
USER_EQ_FUNCS = {}
USER_VARS_SPEC = {}
USER_MODEL_CFG = {}

# === Cache de DataFrames (arquivos grandes) ===
FILE_CACHE = {}

def _stash_df(df: pd.DataFrame) -> dict:
    from uuid import uuid4
    token = str(uuid4())
    FILE_CACHE[token] = df
    return {"__cached__": True, "key": token, "shape": [int(df.shape[0]), int(df.shape[1])]}

def _fetch_df(payload):
    if payload is None:
        return None
    if isinstance(payload, list):
        return pd.DataFrame(payload)
    if isinstance(payload, dict) and payload.get("__cached__"):
        return FILE_CACHE.get(payload.get("key"))
    try:
        return pd.DataFrame(payload)
    except Exception:
        return None

# === Helpers ===
def parse_uploaded(contents: str, filename: str) -> pd.DataFrame:
    if contents is None or filename is None:
        raise ValueError("Nenhum conteúdo/arquivo informado.")

    header, content_string = contents.split(',', 1)
    decoded = base64.b64decode(content_string)
    logger.debug(f"[UPLOAD] filename={filename} base64_header={header[:40]}... bytes={len(decoded):,}")

    if filename.lower().endswith((".xlsx", ".xls")):
        df = pd.read_excel(io.BytesIO(decoded), engine=None)
        logger.debug(f"[UPLOAD] Excel OK shape={df.shape} cols={list(df.columns)[:20]}")
    elif filename.lower().endswith(".csv"):
        s = decoded.decode("utf-8", errors="ignore")
        df = None
        last_err = None
        for sep_try, eng in ((None, "python"), (";", "c"), (",", "c"), ("\t", "c")):
            try:
                df = pd.read_csv(io.StringIO(s), sep=sep_try, engine=eng, on_bad_lines="skip")
                logger.debug(f"[UPLOAD] CSV OK sep={repr(sep_try)} engine={eng} shape={df.shape}")
                break
            except Exception as e:
                last_err = e
                logger.debug(f"[UPLOAD] Falha sep={repr(sep_try)}: {e}")
        if df is None:
            raise ValueError(f"CSV inválido (não foi possível inferir delimitador). Último erro: {last_err}")
    else:
        raise ValueError("Formato não suportado. Use .xlsx ou .csv.")

    if "estrato" in df.columns:
        df["estrato"] = df["estrato"].astype(str).str.strip()

    df = df.reset_index(drop=True)
    if "id" not in df.columns:
        df.insert(0, "id", range(len(df)))
    return df

def add_fit_curve_univar(fig, func, params, df_view, x_col, y_col):
    d = df_view.dropna(subset=[x_col, y_col])
    if d.empty:
        return fig
    xmin, xmax = float(d[x_col].min()), float(d[x_col].max())
    if not np.isfinite(xmin) or not np.isfinite(xmax) or xmin == xmax:
        return fig
    x_grid = np.linspace(xmin, xmax, 200)
    try:
        y_hat = func(x_grid, *params)
        y_hat = np.asarray(y_hat, dtype=float).ravel()
        mask = np.isfinite(x_grid) & np.isfinite(y_hat)
        if mask.sum() < 3:
            fig.add_annotation(text="(fit sem pontos válidos p/ plotar)",
                               xref="paper", yref="paper", x=0.02, y=0.98,
                               showarrow=False, font=dict(color="#888", size=10))
            return fig
        fig.add_trace(go.Scatter(x=x_grid[mask], y=y_hat[mask],
                                 mode="lines", name="Fit",
                                 line=dict(width=3, color="red")))
        fig.data = tuple(list(fig.data[:-1]) + [fig.data[-1]])
    except Exception:
        logger.exception("[PLOT] Falha ao plotar curva univar")
    return fig

def add_fit_curve_generic(fig, func, params, df_view, x_col, y_col, vars_map):
    if df_view is None or not vars_map or x_col is None:
        return fig
    mapped_cols = list(vars_map.values())
    if x_col not in mapped_cols:
        fig.add_annotation(text=f"(Eixo X '{x_col}' não é uma var do modelo. Selecione X ∈ {mapped_cols})",
                           xref="paper", yref="paper", x=0.02, y=0.98,
                           showarrow=False, font=dict(color="#888", size=10))
        return fig
    needed = mapped_cols + ([y_col] if y_col else [])
    d = df_view.dropna(subset=[c for c in needed if c in df_view.columns])
    if d.empty:
        return fig
    xmin, xmax = float(d[x_col].min()), float(d[x_col].max())
    if not np.isfinite(xmin) or not np.isfinite(xmax) or xmin == xmax:
        return fig
    x_grid = np.linspace(xmin, xmax, 200)

    Xg_cols = []
    for _, col in vars_map.items():
        if col == x_col:
            Xg_cols.append(x_grid)
        else:
            Xg_cols.append(np.full_like(x_grid, float(np.nanmedian(d[col]))))
    Xg = np.column_stack(Xg_cols)

    try:
        y_hat = func(Xg, *params)
        y_hat = np.asarray(y_hat, dtype=float).ravel()
        mask = np.isfinite(x_grid) & np.isfinite(y_hat)
        if mask.sum() < 3:
            fig.add_annotation(text="(fit gerou valores não finitos p/ plotar)",
                               xref="paper", yref="paper", x=0.02, y=0.94,
                               showarrow=False, font=dict(color="#888", size=10))
            return fig
        fig.add_trace(go.Scatter(x=x_grid[mask], y=y_hat[mask],
                                 mode="lines", name="Fit",
                                 line=dict(width=3, color="red")))
        fig.data = tuple(list(fig.data[:-1]) + [fig.data[-1]])
    except Exception:
        logger.exception("[PLOT] Falha ao plotar curva (vars)")
    return fig

def regression_metrics(y_true, y_pred):
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    mask = np.isfinite(y_true) & np.isfinite(y_pred)
    y = y_true[mask]; yhat = y_pred[mask]
    if y.size == 0:
        return np.nan, np.nan, np.nan, np.nan
    if y.size > 1 and np.nanstd(y) > 0 and np.nanstd(yhat) > 0:
        r = float(np.corrcoef(y, yhat)[0, 1])
    else:
        r = np.nan
    sse = float(np.sum((y - yhat) ** 2))
    sst = float(np.sum((y - np.mean(y)) ** 2)) if y.size > 1 else np.nan
    r2 = float(1 - sse / sst) if sst and np.isfinite(sst) and sst > 0 else np.nan
    rmse = float(np.sqrt(sse / y.size))
    bias = float(np.mean(yhat - y))
    return r, r2, rmse, bias

def get_initial_guess(name, func, X, y):
    try:
        import inspect
        n_params = max(0, len(inspect.signature(func).parameters) - 1)
    except Exception:
        n_params = 2
    if n_params <= 0:
        return []
    y = np.asarray(y)
    slope = 0.1
    medy = float(np.nanmedian(y)) if np.isfinite(np.nanmedian(y)) else 1.0
    base = [medy, slope]
    while len(base) < n_params:
        base.append(1.0)
    return base[:n_params]

def load_user_models_to_runtime(EQ_FUNCS, VARS_SPEC):
    global USER_EQ_FUNCS, USER_VARS_SPEC, USER_MODEL_CFG
    if not current_user.is_authenticated:
        return []
    models = UserModel.query.filter_by(user_id=current_user.id).all()
    loaded_names = []
    for m in models:
        try:
            param_names = [p.strip() for p in (m.params or "").split(",") if p.strip()]
            if m.kind == "vars":
                var_names = [v.strip() for v in (m.vars_list or "").split(",") if v.strip()]
                f = compile_sympy_vars(m.expr, var_names, param_names)
                USER_VARS_SPEC[m.name] = var_names
                VARS_SPEC[m.name] = var_names
            else:
                f = compile_sympy_univar(m.expr, "x", param_names)
                USER_VARS_SPEC.pop(m.name, None)
                VARS_SPEC.pop(m.name, None)
            USER_EQ_FUNCS[m.name] = f
            USER_MODEL_CFG[m.name] = {
                "solver": m.solver or "trf",
                "maxfev": m.maxfev or 20000,
                "init_kind": m.init_kind or "auto",
                "init_values": [float(v) for v in (m.init_values or "").split(",") if v.strip()]
                               if (m.init_kind == "manual" and m.init_values) else None
            }
            loaded_names.append(m.name)
            EQ_FUNCS[m.name] = f
        except Exception:
            logger.exception(f"[MODELS] falha carregando modelo '{m.name}'")
    return loaded_names

# === Registro de Callbacks ===
def register_callbacks(app, context):
    # (0) Upload
    @app.callback(
        Output("orig-data", "data"),
        Output("store-num-cols", "data"),
        Output("store-all-cols", "data"),
        Output("status", "children"),
        Input("file-upload", "contents"),
        State("file-upload", "filename"),
        prevent_initial_call=True
    )
    def on_file_uploaded(contents, filename):
        try:
            logger.debug(f"[CB] on_file_uploaded: filename={filename}, has? {contents is not None}")
            df = parse_uploaded(contents, filename)

            num_cols = [c for c in df.select_dtypes(include="number").columns if c != "id"]
            all_cols = [c for c in df.columns if c != "id"]

            approx_bytes = int(df.memory_usage(deep=True).sum())
            is_big = (len(df) > 100_000) or (approx_bytes > 5_000_000)

            if is_big:
                orig_payload = _stash_df(df)
                size_msg = f" | Modo grande (cache servidor) ~{approx_bytes/1e6:.1f} MB"
            else:
                orig_payload = df.to_dict("records")
                size_msg = ""

            msg = f"Arquivo carregado: {filename}\nLinhas: {len(df)} | Colunas: {len(df.columns)}{size_msg}"
            return (orig_payload, num_cols, all_cols, msg)
        except Exception as e:
            logger.exception("[CB] Falha no upload")
            short = f"Falha ao ler arquivo: {e.__class__.__name__}: {e}"
            hint = "\nDicas: confirme .xlsx/.csv UTF-8; verifique separador (; ou ,); teste um arquivo menor."
            return (no_update, no_update, no_update, short + hint)

    # (A) Vars mapping dinâmico
    @app.callback(
        Output("vars-mapping", "children"),
        Input("eq-dd", "value"),
        State("store-all-cols", "data")
    )
    def render_vars_mapping(eq_name, all_cols):
        if not all_cols:
            return [html.Div("Carregue um arquivo para habilitar mapeamento.", style={"color": "#666"})]
        vars_needed = context["VARS_SPEC"].get(eq_name, [])
        if not vars_needed:
            return [html.Div("Esta equação não requer mapeamento de variáveis adicionais.", style={"color": "#666"})]
        opts = [{"label": c, "value": c} for c in all_cols]
        return [
            html.Div([
                html.Small(f"{var} (coluna)"),
                dcc.Dropdown(
                    id={"role": "var-map", "name": var},
                    options=opts,
                    value=None,
                    clearable=True,
                    style={"minWidth": 220}
                )
            ]) for var in vars_needed
        ]

    # (B) Opções dos dropdowns (inclui estrato-col-dd agora que ele existe no layout)
    @app.callback(
        Output("x-dd", "options"),
        Output("y-dd", "options"),
        Output("map-target", "options"),
        Output("estrato-col-dd", "options"),
        Input("store-num-cols", "data"),
        Input("store-all-cols", "data"),
        prevent_initial_call=True
    )
    def refresh_dropdown_options(num_cols, all_cols):
        x_opts = [{"label": c, "value": c} for c in (num_cols or [])]
        y_opts = [{"label": c, "value": c} for c in (num_cols or [])]
        all_opts = [{"label": c, "value": c} for c in (all_cols or [])]
        return x_opts, y_opts, all_opts, all_opts

    # (C) Valores padrão de X/Y
    @app.callback(
        Output("x-dd", "value"),
        Output("y-dd", "value"),
        Input("store-num-cols", "data"),
        prevent_initial_call=True
    )
    def set_default_axes(num_cols):
        if not num_cols:
            return None, None
        default_x = num_cols[0]
        default_y = num_cols[1] if len(num_cols) > 1 else num_cols[0]
        return default_x, default_y

    # (D) Mostrar/Ocultar campos de estratificação (sem criar componentes novos)
    @app.callback(
        Output("estrato-col-wrap", "style"),
        Output("estrato-val-wrap", "style"),
        Input("strat-check", "value"),
        prevent_initial_call=False
    )
    def toggle_strat_fields(strat_value):
        on = isinstance(strat_value, list) and ("on" in strat_value)
        style = {"display": "block"} if on else {"display": "none"}
        return style, style

    # (E) Opções do valor do estrato
    @app.callback(
        Output("estrato-val-dd", "options"),
        Output("estrato-val-dd", "value"),
        Input("estrato-col-dd", "value"),
        Input("base-data", "data"),
        prevent_initial_call=True
    )
    def update_estrato_values(estrato_col, base_data):
        if base_data is None or not estrato_col:
            return [], None
        df_base = _fetch_df(base_data)
        if df_base is None or estrato_col not in df_base.columns:
            return [], None
        vals = df_base[estrato_col].dropna().unique().tolist()
        vals = sorted(vals, key=lambda x: str(x))
        opts = [{"label": str(v), "value": v} for v in vals]
        value = vals[0] if vals else None
        return opts, value

    # (F) Gráfico
    @app.callback(
        Output("scatter-plot", "figure"),
        Input("strat-check", "value"),
        Input("estrato-col-dd", "value"),
        Input("estrato-val-dd", "value"),
        Input("x-dd", "value"),
        Input("y-dd", "value"),
        Input("base-data", "data"),
        Input("fit-state", "data"),
    )
    def update_graph(strat_value, estrato_col, estrato_val, x_col, y_col, base_data, fit_state):
        if base_data is None or x_col is None or y_col is None:
            return {}
        df_base = _fetch_df(base_data)
        if df_base is None:
            return {}

        strat_on = isinstance(strat_value, list) and ("on" in strat_value)
        if strat_on and estrato_col and (estrato_col in df_base.columns) and (estrato_val is not None):
            df_view = df_base[df_base[estrato_col] == estrato_val]
        else:
            df_view = df_base

        d_plot = df_view.dropna(subset=[x_col, y_col])
        fig = px.scatter(d_plot, x=x_col, y=y_col, custom_data=["id"])
        fig.update_traces(mode="markers", marker={"opacity": 0.6, "size": 7}, selector=dict(type="scatter"))

        if fit_state and isinstance(fit_state, dict):
            name = fit_state.get("name")
            params = fit_state.get("params")
            uses_vars = fit_state.get("uses_vars", False)
            func = context["EQ_FUNCS"].get(name)
            if func and params:
                if uses_vars:
                    vars_map = fit_state.get("maps") or {}
                    fig = add_fit_curve_generic(fig, func, params, df_view, x_col, y_col, vars_map)
                else:
                    fig = add_fit_curve_univar(fig, func, params, df_view, x_col, y_col)

        # pontos primeiro, linhas por último
        if len(fig.data) > 1:
            points, lines = [], []
            for tr in fig.data:
                if getattr(tr, "mode", None) and "lines" in tr.mode:
                    lines.append(tr)
                else:
                    points.append(tr)
            fig.data = tuple(points + lines)
            for tr in lines:
                if not getattr(tr, "line", None):
                    tr.line = {}
                tr.line["width"] = 3

        fig.update_layout(dragmode="lasso")
        return fig

    # (G) Ações
    @app.callback(
        Output("base-data", "data"),
        Output("removed-ids", "data"),
        Output("output", "children"),
        Output("fit-state", "data"),
        Input("orig-data", "data"),
        Input("remove-btn", "n_clicks"),
        Input("reset-btn", "n_clicks"),
        Input("show-removed-btn", "n_clicks"),
        Input("fit-btn", "n_clicks"),
        State("strat-check", "value"),
        State("estrato-col-dd", "value"),
        State("estrato-val-dd", "value"),
        State("scatter-plot", "selectedData"),
        State("x-dd", "value"),
        State("y-dd", "value"),
        State("eq-dd", "value"),
        State({"role": "var-map", "name": ALL}, "id"),
        State({"role": "var-map", "name": ALL}, "value"),
        State("map-target", "value"),
        State("base-data", "data"),
        State("removed-ids", "data"),
        State("orig-data", "data"),
        prevent_initial_call=True
    )
    def handle_actions(orig_data_trigger,
                       n_remove, n_reset, n_show, n_fit,
                       strat_value, estrato_col, estrato_val,
                       selectedData, x_col, y_col, eq_name,
                       var_ids, var_vals, map_target,
                       base_data, removed_ids, orig_data):

        trig = ctx.triggered_id

        # Inicializa após upload
        if trig == "orig-data":
            if orig_data is None:
                return no_update, no_update, no_update, no_update
            return orig_data, [], no_update, None

        if orig_data is None:
            return no_update, no_update, "Carregue um arquivo primeiro.", no_update

        df_base = _fetch_df(base_data) if base_data is not None else _fetch_df(orig_data)
        if df_base is None:
            return no_update, no_update, "Falha ao acessar a base (cache nulo).", no_update

        strat_on = isinstance(strat_value, list) and ("on" in strat_value)

        def apply_strat(dfin: pd.DataFrame) -> pd.DataFrame:
            if strat_on and estrato_col and (estrato_col in dfin.columns) and (estrato_val is not None):
                return dfin[dfin[estrato_col] == estrato_val]
            return dfin

        # Reset
        if trig == "reset-btn":
            return orig_data, [], "Base resetada; IDs removidos zerados.", None

        # Mostrar IDs removidos
        if trig == "show-removed-btn":
            if not removed_ids:
                return no_update, no_update, "Nenhum ID removido até agora.", no_update
            return no_update, no_update, f"IDs removidos acumulados: {sorted(set(removed_ids))}", no_update

        # Remover selecionados
        if trig == "remove-btn":
            if not selectedData or "points" not in selectedData:
                return base_data, removed_ids, "Nenhum ponto selecionado.", no_update
            picked_ids = []
            for p in selectedData["points"]:
                cd = p.get("customdata")
                if cd is not None:
                    pid = cd[0] if isinstance(cd, (list, tuple)) else cd
                    try:
                        picked_ids.append(int(pid))
                    except Exception:
                        pass
            if not picked_ids:
                if x_col is None or y_col is None:
                    return base_data, removed_ids, "Defina Eixos X/Y.", no_update
                d_view = apply_strat(df_base).dropna(subset=[x_col, y_col]).reset_index(drop=True)
                if len(d_view) == 0:
                    return base_data, removed_ids, "Nenhuma linha válida para mapear seleção.", no_update
                idxs = [int(p["pointIndex"]) for p in selectedData["points"]]
                picked_ids = d_view.loc[idxs, "id"].tolist()
            if not picked_ids:
                return base_data, removed_ids, "Nenhum ID capturado.", no_update

            df_new = df_base[~df_base["id"].isin(picked_ids)]
            removed_ids_new = (removed_ids or []) + picked_ids

            approx_bytes = int(df_new.memory_usage(deep=True).sum())
            is_big = (len(df_new) > 100_000) or (approx_bytes > 5_000_000)
            new_payload = _stash_df(df_new) if is_big or (isinstance(base_data, dict) and base_data.get("__cached__")) \
                          else df_new.to_dict("records")

            return new_payload, removed_ids_new, f"Removidos {len(picked_ids)}: {picked_ids}", no_update

        # Ajuste (fit)
        if trig == "fit-btn":
            if not eq_name:
                return no_update, no_update, "Selecione uma equação em Eq.py.", no_update
            func = context["EQ_FUNCS"].get(eq_name)
            if func is None:
                return no_update, no_update, f"Equação '{eq_name}' não encontrada.", no_update

            df_view = apply_strat(df_base)

            vars_needed = context["VARS_SPEC"].get(eq_name, [])
            vars_map = {}
            if vars_needed:
                for comp_id, val in zip(var_ids, var_vals):
                    if isinstance(comp_id, dict) and comp_id.get("role") == "var-map":
                        vars_map[comp_id.get("name")] = val

            uses_vars = bool(vars_needed)

            cfg = USER_MODEL_CFG.get(eq_name, {})
            method = cfg.get("solver", None)        # trf | dogbox | lm
            maxfev = int(cfg.get("maxfev", 20000))
            p0_manual = cfg.get("init_values", None)
            kwargs = {"maxfev": maxfev}
            if method:
                kwargs["method"] = method

            if uses_vars:
                missing = [v for v in vars_needed if not vars_map.get(v)]
                if missing:
                    return no_update, no_update, f"Faltam mapeamentos: {missing}", no_update
                if not map_target or map_target not in df_view.columns:
                    return no_update, no_update, "Selecione a coluna alvo (Target Y).", no_update

                cols_need = [vars_map[v] for v in vars_needed] + [map_target]
                df_fit = df_view.dropna(subset=cols_need)
                if df_fit.empty:
                    return no_update, no_update, "Sem dados (NaN) após mapeamento para ajuste.", no_update

                X = df_fit[[vars_map[v] for v in vars_needed]].to_numpy(dtype=float, copy=True)
                y = df_fit[map_target].to_numpy(dtype=float, copy=True)

                p0 = p0_manual if (p0_manual and len(p0_manual) > 0) else get_initial_guess(eq_name, func, X, y)
                try:
                    logger.debug(f"[FIT] eq={eq_name} uses_vars=True X.shape={X.shape} y.shape={y.shape}")
                    popt, _ = curve_fit(func, X, y, p0=p0, **kwargs)
                    y_hat = func(X, *popt)
                    r, r2, rmse, bias = regression_metrics(y, y_hat)
                    fit_info = {"name": eq_name, "params": [float(v) for v in popt],
                                "uses_vars": True, "maps": vars_map, "target": map_target}
                    msg = (f"Ajuste OK: {eq_name}\n"
                           f"Parâmetros: {np.round(fit_info['params'], 6).tolist()}\n"
                           f"r={r:.4f} | r²={r2:.4f} | RMSE={rmse:.4f} | Bias={bias:.4f}")
                    return no_update, no_update, msg, fit_info
                except Exception as e:
                    logger.exception(f"[FIT] Falha no ajuste (vars) eq={eq_name}")
                    return no_update, no_update, f"Falha no ajuste: {e}", no_update
            else:
                if not x_col or not y_col:
                    return no_update, no_update, "Defina Eixo X e Eixo Y para ajuste univariado.", no_update
                df_fit = df_view.dropna(subset=[x_col, y_col])
                if df_fit.empty:
                    return no_update, no_update, "Sem dados válidos (NaN) para ajuste univariado.", no_update

                X = df_fit[x_col].to_numpy(dtype=float, copy=True)
                y = df_fit[y_col].to_numpy(dtype=float, copy=True)

                p0 = p0_manual if (p0_manual and len(p0_manual) > 0) else get_initial_guess(eq_name, func, X, y)
                try:
                    logger.debug(f"[FIT] eq={eq_name} uses_vars=False X.len={len(X)} y.shape={y.shape}")
                    popt, _ = curve_fit(func, X, y, p0=p0, **kwargs)
                    y_hat = func(X, *popt)
                    r, r2, rmse, bias = regression_metrics(y, y_hat)
                    fit_info = {"name": eq_name, "params": [float(v) for v in popt], "uses_vars": False}
                    msg = (f"Ajuste OK: {eq_name}\n"
                           f"Parâmetros: {np.round(fit_info['params'], 6).tolist()}\n"
                           f"r={r:.4f} | r²={r2:.4f} | RMSE={rmse:.4f} | Bias={bias:.4f}")
                    return no_update, no_update, msg, fit_info
                except Exception as e:
                    logger.exception(f"[FIT] Falha no ajuste (uni) eq={eq_name}")
                    return no_update, no_update, f"Falha no ajuste: {e}", no_update

        return no_update, no_update, no_update, no_update

    # Modal abrir/fechar
    @app.callback(
        Output("new-model-wrap", "style"),
        Input("open-new-model", "n_clicks"),
        Input("nm-cancel", "n_clicks"),
        prevent_initial_call=True
    )
    def toggle_new_model_panel(n_open, n_close):
        trig = ctx.triggered_id
        if trig == "open-new-model":
            return {"display": "block"}
        return {"display": "none"}

    # Campos dinâmicos (vars / init manual)
    @app.callback(
        Output("nm-vars-spec-wrap", "style"),
        Output("nm-init-values-wrap", "style"),
        Input("nm-kind", "value"),
        Input("nm-init-kind", "value"),
        prevent_initial_call=False
    )
    def toggle_new_model_fields(kind, init_kind):
        vars_style = {"display": "block"} if kind == "vars" else {"display": "none"}
        init_style = {"display": "block"} if init_kind == "manual" else {"display": "none"}
        return vars_style, init_style

    # Preload modelos do usuário
    @app.callback(
        Output("eq-dd", "options"),
        Input("url", "pathname")
    )
    def preload_user_models(_):
        EQ_FUNCS = context["EQ_FUNCS"]
        VARS_SPEC = context["VARS_SPEC"]
        load_user_models_to_runtime(EQ_FUNCS, VARS_SPEC)
        eq_names = sorted(EQ_FUNCS.keys())
        return [{"label": n, "value": n} for n in eq_names]

    # Validar / Salvar modelo
    @app.callback(
        Output("nm-status", "children"),
        Output("eq-dd", "options", allow_duplicate=True),
        Output("eq-dd", "value"),
        Output("new-model-wrap", "style", allow_duplicate=True),
        Input("nm-validate", "n_clicks"),
        Input("nm-save", "n_clicks"),
        State("nm-name", "value"),
        State("nm-kind", "value"),
        State("nm-expr", "value"),
        State("nm-params", "value"),
        State("nm-init-kind", "value"),
        State("nm-init-values", "value"),  # <-- agora existe sempre
        State("nm-solver", "value"),
        State("nm-maxfev", "value"),
        State("nm-vars-spec", "value"),  # <-- agora existe sempre
        State("base-data", "data"),
        prevent_initial_call=True
    )
    def validate_or_save(nv, ns, nm_name, nm_kind, nm_expr, nm_params,
                         nm_init_kind, nm_init_values, nm_solver, nm_maxfev,
                         nm_vars_list, base_data):
        trig = ctx.triggered_id
        name = (nm_name or "").strip()
        if not name:
            return ("Informe um nome para o modelo.", no_update, no_update, no_update)

        try:
            param_names = [p.strip() for p in (nm_params or "").split(",") if p.strip()]
            if nm_kind == "vars":
                var_names = [v.strip() for v in (nm_vars_list or "").split(",") if v.strip()]
                f_compiled = compile_sympy_vars(nm_expr, var_names, param_names)
            else:
                var_names = None
                f_compiled = compile_sympy_univar(nm_expr, "x", param_names)
        except Exception as e:
            return (f"Erro de sintaxe ao compilar: {e}", no_update, no_update, no_update)

        # validação rápida com base atual (se houver)
        try:
            if base_data:
                df = _fetch_df(base_data)
                if df is not None:
                    if nm_kind == "vars" and var_names and set(var_names).issubset(df.columns):
                        d = df.dropna(subset=var_names)
                        if len(d) >= 3:
                            X_try = d[var_names].head(5).to_numpy()
                            p_try = np.ones(len(param_names))
                            _ = f_compiled(X_try, *p_try)
                    elif nm_kind == "uni":
                        num_cols = df.select_dtypes(include="number").columns.tolist()
                        if num_cols:
                            x_try = df[num_cols[0]].dropna().head(5).to_numpy()
                            p_try = np.ones(len(param_names))
                            _ = f_compiled(x_try, *p_try)
            ok_msg = "✅ Validação OK."
        except Exception as e:
            return (f"Falha na validação com dados: {e}", no_update, no_update, no_update)

        if trig == "nm-validate":
            return (ok_msg, no_update, no_update, no_update)

        # salvar no banco
        if not current_user.is_authenticated:
            return ("Faça login para salvar modelos.", no_update, no_update, no_update)

        init_values_txt = None
        if nm_init_kind == "manual" and (nm_init_values or "").strip():
            init_values_txt = ",".join([v.strip() for v in nm_init_values.split(",") if v.strip()])

        m = UserModel.query.filter_by(user_id=current_user.id, name=name).one_or_none()
        if m is None:
            m = UserModel(user_id=current_user.id, name=name)
            db.session.add(m)

        m.kind = nm_kind
        m.expr = (nm_expr or "").strip()
        m.params = ",".join(param_names)
        m.vars_list = ",".join(var_names) if (nm_kind == "vars" and var_names) else None
        m.init_kind = nm_init_kind
        m.init_values = init_values_txt
        m.solver = nm_solver or "trf"
        m.maxfev = int(nm_maxfev or 20000)
        db.session.commit()

        EQ_FUNCS = context["EQ_FUNCS"]
        VARS_SPEC = context["VARS_SPEC"]

        if nm_kind == "vars":
            USER_VARS_SPEC[name] = var_names
            VARS_SPEC[name] = var_names
        else:
            USER_VARS_SPEC.pop(name, None)
            VARS_SPEC.pop(name, None)

        USER_EQ_FUNCS[name] = f_compiled
        USER_MODEL_CFG[name] = {
            "solver": m.solver, "maxfev": m.maxfev,
            "init_kind": m.init_kind,
            "init_values": [float(v) for v in (init_values_txt or "").split(",") if v.strip()]
                           if m.init_kind == "manual" else None
        }
        EQ_FUNCS[name] = f_compiled

        eq_names_all = sorted(EQ_FUNCS.keys())
        options = [{"label": n, "value": n} for n in eq_names_all]

        return (f"✅ Modelo '{name}' salvo.", options, name, no_update)
