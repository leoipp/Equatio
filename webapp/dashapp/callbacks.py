import base64, io
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from dash import Input, Output, State, ctx, no_update, dcc, html
from dash.dependencies import ALL
from scipy.optimize import curve_fit


# =========================
# Helpers
# =========================
def parse_uploaded(contents: str, filename: str) -> pd.DataFrame:
    """Lê .xlsx ou .csv do dcc.Upload (base64) e retorna DataFrame com 'id' e 'estrato' (se necessário)."""
    if contents is None or filename is None:
        return None
    _, content_string = contents.split(',')
    decoded = base64.b64decode(content_string)

    if filename.lower().endswith((".xlsx", ".xls")):
        df = pd.read_excel(io.BytesIO(decoded))
    elif filename.lower().endswith(".csv"):
        s = decoded.decode("utf-8", errors="ignore")
        try:
            df = pd.read_csv(io.StringIO(s), sep=None, engine="python")
        except Exception:
            try:
                df = pd.read_csv(io.StringIO(s), sep=";")
            except Exception:
                df = pd.read_csv(io.StringIO(s), sep=",")
    else:
        raise ValueError("Formato não suportado. Use .xlsx ou .csv.")

    # normalizações
    if "estrato" in df.columns:
        df["estrato"] = df["estrato"].astype(str).str.strip()

    df = df.reset_index(drop=True)
    df.insert(0, "id", range(len(df)))
    return df


def make_scatter(dframe: pd.DataFrame, x_col: str, y_col: str):
    if dframe is None or x_col is None or y_col is None:
        return {}
    if x_col not in dframe.columns or y_col not in dframe.columns:
        return {}
    d_plot = dframe.dropna(subset=[x_col, y_col])
    fig = px.scatter(d_plot, x=x_col, y=y_col, custom_data=["id"])
    fig.update_layout(dragmode="lasso")
    return fig


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
            # opcional: adicionar anotação leve
            fig.add_annotation(text="(fit sem pontos válidos p/ plotar)", xref="paper", yref="paper",
                               x=0.02, y=0.98, showarrow=False, font=dict(color="#888", size=10))
            return fig
        fig.add_trace(
            go.Scatter(
                x=x_grid[mask],
                y=y_hat[mask],
                mode="lines",
                name="Fit",
                line=dict(width=3, color="red"),  # opcional: destaque
            ),
            row=None, col=None
        )

        # Garante que a curva fique por cima
        fig.data = tuple(list(fig.data[:-1]) + [fig.data[-1]])
    except Exception as e:
        print(f"[WARN] Falha ao plotar curva univar: {e}")
    return fig


def add_fit_curve_generic(fig, func, params, df_view, x_col, y_col, vars_map):
    if df_view is None or not vars_map or x_col is None:
        return fig

    # o X do gráfico precisa ser uma das variáveis mapeadas
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

    # monta grade: var do eixo X varia, demais ficam na mediana
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
        fig.add_trace(
            go.Scatter(
                x=x_grid[mask],
                y=y_hat[mask],
                mode="lines",
                name="Fit",
                line=dict(width=3, color="red"),  # opcional: destaque
            ),
            row=None, col=None
        )

        # Garante que a curva fique por cima
        fig.data = tuple(list(fig.data[:-1]) + [fig.data[-1]])

    except Exception as e:
        print(f"[WARN] Falha ao plotar curva (vars): {e}")
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
        sig = inspect.signature(func)
        n_params = max(0, len(sig.parameters) - 1)
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


# =========================
# Registro de Callbacks
# =========================
def register_callbacks(app, context):
    EQ_FUNCS = context["EQ_FUNCS"]
    VARS_SPEC = context["VARS_SPEC"]

    # (0) Upload -> apenas Stores + status
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
            df = parse_uploaded(contents, filename)
        except Exception as e:
            return (no_update, no_update, no_update, f"Falha ao ler arquivo: {e}")

        num_cols = [c for c in df.select_dtypes(include="number").columns if c != "id"]
        all_cols = [c for c in df.columns if c != "id"]

        msg = (
            f"Arquivo carregado: {filename}\n"
            f"Linhas: {len(df)} | Colunas: {len(df.columns)}"
        )

        # Apenas orig-data + stores. Quem inicializa base/removed/fit é o callback de ações.
        return (
            df.to_dict("records"),   # orig-data
            num_cols,                # store-num-cols
            all_cols,                # store-all-cols
            msg                      # status
        )

    # (A) Mapeamento dinâmico conforme equação
    @app.callback(
        Output("vars-mapping", "children"),
        Input("eq-dd", "value"),
        State("store-all-cols", "data")
    )
    def render_vars_mapping(eq_name, all_cols):
        if not all_cols:
            return [html.Div("Carregue um arquivo para habilitar mapeamento.", style={"color": "#666"})]
        vars_needed = VARS_SPEC.get(eq_name, [])
        if not vars_needed:
            return [html.Div("Esta equação não requer mapeamento de variáveis adicionais.", style={"color": "#666"})]
        children = []
        for var in vars_needed:
            children.append(
                html.Div([
                    html.Small(f"{var} (coluna)"),
                    dcc.Dropdown(
                        id={"role": "var-map", "name": var},
                        options=[{"label": c, "value": c} for c in all_cols],
                        value=None,
                        clearable=True,
                        style={"minWidth": 220}
                    )
                ])
            )
        return children

    # (B) Atualiza opções dos dropdowns gerais (X/Y/Target e coluna de estratos)
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
        target_opts = [{"label": c, "value": c} for c in (all_cols or [])]
        estrato_col_opts = [{"label": c, "value": c} for c in (all_cols or [])]
        return x_opts, y_opts, target_opts, estrato_col_opts

    # (C) Define valores padrão de X/Y quando as colunas numéricas mudam
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

    # (D) Habilita/Desabilita os dropdowns de estratos conforme o check
    @app.callback(
        Output("estrato-col-dd", "disabled"),
        Output("estrato-val-dd", "disabled"),
        Input("strat-check", "value")
    )
    def toggle_estrato_dd(strat_value):
        enabled = isinstance(strat_value, list) and ("on" in strat_value)
        disabled = not enabled
        return disabled, disabled

    # (E) Atualiza as opções do valor do estrato conforme a coluna escolhida + base de dados
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
        df_base = pd.DataFrame(base_data)
        if estrato_col not in df_base.columns:
            return [], None
        vals = df_base[estrato_col].dropna().unique().tolist()
        vals = sorted(vals, key=lambda x: str(x))
        opts = [{"label": str(v), "value": v} for v in vals]
        value = vals[0] if vals else None
        return opts, value

    # (F) Atualiza gráfico com base em filtro/eixos/base/fit
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
        df_base = pd.DataFrame(base_data)

        # aplica filtro de estratificação
        strat_on = isinstance(strat_value, list) and ("on" in strat_value)
        if strat_on and estrato_col and (estrato_col in df_base.columns) and (estrato_val is not None):
            df_view = df_base[df_base[estrato_col] == estrato_val]
        else:
            df_view = df_base

        # 1) pontos
        d_plot = df_view.dropna(subset=[x_col, y_col])
        fig = px.scatter(d_plot, x=x_col, y=y_col, custom_data=["id"])
        # deixa os pontos um pouco transparentes
        fig.update_traces(mode="markers", marker={"opacity": 0.6, "size": 7}, selector=dict(type="scatter"))

        # 2) curva
        if fit_state and isinstance(fit_state, dict):
            name = fit_state.get("name")
            params = fit_state.get("params")
            uses_vars = fit_state.get("uses_vars", False)
            func = context["EQ_FUNCS"].get(name)
            if func and params:
                if uses_vars:
                    vars_map = fit_state.get("maps") or {}
                    # --- sua helper ---
                    fig = add_fit_curve_generic(fig, func, params, df_view, x_col, y_col, vars_map)
                else:
                    # --- sua helper ---
                    fig = add_fit_curve_univar(fig, func, params, df_view, x_col, y_col)

        # 3) REORDENA: garanta que todas as linhas (lines) fiquem por cima
        if len(fig.data) > 1:
            points = []
            lines = []
            for tr in fig.data:
                # qualquer trace com 'lines' no modo vai pro topo
                if hasattr(tr, "mode") and tr.mode and "lines" in tr.mode:
                    lines.append(tr)
                else:
                    points.append(tr)
            # remonta: pontos primeiro, linhas por último
            fig.data = tuple(points + lines)

            # destaque da linha (ajusta todas que são 'lines')
            for tr in lines:
                if not getattr(tr, "line", None):
                    tr.line = {}
                tr.line["width"] = 3
                # cor opcional (remova se quiser manter automático)
                # tr.line["color"] = "red"

        fig.update_layout(dragmode="lasso")
        return fig

    # (G) Remover / Resetar / Mostrar IDs / Ajustar
    #     -> ÚNICO callback que escreve em base-data, removed-ids e fit-state
    @app.callback(
        Output("base-data", "data"),
        Output("removed-ids", "data"),
        Output("output", "children"),
        Output("fit-state", "data"),
        Input("orig-data", "data"),          # inicializa
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

        # Inicialização (após upload): somente este callback escreve base/removed/fit
        if trig == "orig-data":
            if orig_data is None:
                return no_update, no_update, no_update, no_update
            return orig_data, [], no_update, None

        # Sem dados carregados
        if orig_data is None:
            return no_update, no_update, "Carregue um arquivo primeiro.", no_update

        df_base = pd.DataFrame(base_data) if base_data is not None else pd.DataFrame(orig_data)

        # helper de filtro por estratificação
        strat_on = isinstance(strat_value, list) and ("on" in strat_value)
        def apply_strat(dfin: pd.DataFrame) -> pd.DataFrame:
            if strat_on and estrato_col and (estrato_col in dfin.columns) and (estrato_val is not None):
                return dfin[dfin[estrato_col] == estrato_val]
            return dfin

        # Reset base
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
                idxs = [int(p["pointIndex"]) for p in selectedData["points"]]
                if len(d_view) == 0:
                    return base_data, removed_ids, "Nenhuma linha válida para mapear seleção.", no_update
                picked_ids = d_view.loc[idxs, "id"].tolist()
            if not picked_ids:
                return base_data, removed_ids, "Nenhum ID capturado.", no_update

            df_new = df_base[~df_base["id"].isin(picked_ids)]
            removed_ids_new = (removed_ids or []) + picked_ids
            return df_new.to_dict("records"), removed_ids_new, f"Removidos {len(picked_ids)}: {picked_ids}", no_update

        # Ajustar equação
        if trig == "fit-btn":
            if not eq_name:
                return no_update, no_update, "Selecione uma equação em Eq.py.", no_update
            func = context["EQ_FUNCS"].get(eq_name)
            if func is None:
                return no_update, no_update, f"Equação '{eq_name}' não encontrada.", no_update

            df_view = apply_strat(df_base)

            # Mapeamentos dinâmicos (se a equação usa Vars)
            vars_needed = context["VARS_SPEC"].get(eq_name, [])
            vars_map = {}
            if vars_needed:
                for comp_id, val in zip(var_ids, var_vals):
                    if isinstance(comp_id, dict) and comp_id.get("role") == "var-map":
                        vars_map[comp_id.get("name")] = val

            uses_vars = bool(vars_needed)

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

                p0 = get_initial_guess(eq_name, func, X, y)
                try:
                    popt, _ = curve_fit(func, X, y, p0=p0, maxfev=20000)
                    y_hat = func(X, *popt)
                    r, r2, rmse, bias = regression_metrics(y, y_hat)
                    fit_info = {
                        "name": eq_name,
                        "params": [float(v) for v in popt],
                        "uses_vars": True,
                        "maps": vars_map,
                        "target": map_target,
                    }
                    msg = (
                        f"Ajuste OK: {eq_name}\n"
                        f"Parâmetros: {np.round(fit_info['params'], 6).tolist()}\n"
                        f"r={r:.4f} | r²={r2:.4f} | RMSE={rmse:.4f} | Bias={bias:.4f}"
                    )
                    return no_update, no_update, msg, fit_info
                except Exception as e:
                    return no_update, no_update, f"Falha no ajuste: {e}", no_update

            # Caso univariado
            else:
                if not x_col or not y_col:
                    return no_update, no_update, "Defina Eixo X e Eixo Y para ajuste univariado.", no_update
                df_fit = df_view.dropna(subset=[x_col, y_col])
                if df_fit.empty:
                    return no_update, no_update, "Sem dados válidos (NaN) para ajuste univariado.", no_update

                X = df_fit[x_col].to_numpy(dtype=float, copy=True)
                y = df_fit[y_col].to_numpy(dtype=float, copy=True)

                p0 = get_initial_guess(eq_name, func, X, y)
                try:
                    popt, _ = curve_fit(func, X, y, p0=p0, maxfev=20000)
                    y_hat = func(X, *popt)
                    r, r2, rmse, bias = regression_metrics(y, y_hat)
                    fit_info = {"name": eq_name, "params": [float(v) for v in popt], "uses_vars": False}
                    msg = (
                        f"Ajuste OK: {eq_name}\n"
                        f"Parâmetros: {np.round(fit_info['params'], 6).tolist()}\n"
                        f"r={r:.4f} | r²={r2:.4f} | RMSE={rmse:.4f} | Bias={bias:.4f}"
                    )
                    return no_update, no_update, msg, fit_info
                except Exception as e:
                    return no_update, no_update, f"Falha no ajuste: {e}", no_update

        # Sem ação tratada
        return no_update, no_update, no_update, no_update
