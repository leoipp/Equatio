# layout.py
from dash import dcc, html

def build_layout(EQ_FUNCS, VARS_SPEC):
    eq_names = sorted(EQ_FUNCS.keys())

    return html.Div(className="container", children=[
        # ===== Stores =====
        dcc.Store(id="orig-data", data=None),
        dcc.Store(id="base-data", data=None),
        dcc.Store(id="removed-ids", data=[]),
        dcc.Store(id="fit-state", data=None),
        dcc.Store(id="store-num-cols", data=[]),
        dcc.Store(id="store-all-cols", data=[]),

        # Necessário para callbacks que dependem de URL (ex.: preload_user_models)
        dcc.Location(id="url"),

        # ===== Layout 2 colunas =====
        html.Div(className="dash-layout", children=[

            # -------- Sidebar (Esquerda) --------
            html.Div(className="dash-sidebar", children=[

                # 1) Upload
                html.Div(className="panel", children=[
                    html.H3("Dados", className="panel-title"),
                    html.Div(className="row", children=[
                        dcc.Upload(
                            id="file-upload",
                            children=html.Button("Selecionar arquivo", className="btn"),
                            multiple=False,
                            className="upload-wrap",
                        ),
                        html.Small("Selecione um .xlsx ou .csv",
                                   style={"marginLeft": "8px", "color": "#666"}),
                    ]),
                ]),

                # 2) Estratificação (checkbox + campos já no layout, só ocultos/exibidos)
                html.Div(className="panel mt-14", children=[
                    html.Label("Estratificação"),
                    dcc.Checklist(
                        id="strat-check",
                        options=[{"label": " habilitar", "value": "on"}],
                        value=[],
                        style={"marginTop": "6px"}
                    ),

                    # campos fixos com WRAPPER (esconde label + dropdown juntos)
                    html.Div(id="strat-fields", className="mt-8", children=[
                        html.Div(id="estrato-col-wrap", className="mt-8", children=[
                            html.Small("Coluna de estratos"),
                            dcc.Dropdown(
                                id="estrato-col-dd",
                                options=[], value=None,
                                clearable=True, className="minw-220",
                            ),
                        ]),
                        html.Div(id="estrato-val-wrap", className="mt-8", children=[
                            html.Small("Valor do estrato"),
                            dcc.Dropdown(
                                id="estrato-val-dd",
                                options=[], value=None,
                                clearable=True, className="minw-220",
                            ),
                        ]),
                    ]),
                ]),

                # 3) Equação (Eq.py) — X/Y foram movidos para a direita
                html.Div(className="panel mt-14", children=[
                    html.H3("Equação (Eq.py)", className="panel-title"),
                    html.Div(className="row", style={"display": "flex", "gap": "16px", "flexWrap": "wrap"}, children=[
                        html.Div(className="minw-260", children=[
                            dcc.Dropdown(
                                id="eq-dd",
                                options=[{"label": n, "value": n} for n in eq_names],
                                value=(eq_names[0] if eq_names else None),
                                clearable=True,
                                className="minw-260",
                            ),
                            html.Button("➕ Cadastrar modelo", id="open-new-model", n_clicks=0, className="btn mt-8"),
                        ]),
                    ]),
                ]),

                # Modal (controlada por classes .modal / .modal.show)
                html.Div(
                    id="new-model-wrap",
                    className="panel mt-14",
                    style={"display": "none"},  # só aparece quando clicar no botão
                    children=[
                        html.Div(style={"display": "flex", "alignItems": "center", "gap": "12px",
                                        "justifyContent": "space-between"}, children=[
                            html.H3("Cadastrar novo modelo", className="panel-title", style={"margin": 0}),
                            html.Button("×", id="nm-cancel", n_clicks=0, className="btn"),  # fecha o painel
                        ]),

                        html.Label("Nome do modelo"),
                        dcc.Input(id="nm-name", type="text",
                                  placeholder="Ex.: Exponencial", className="input"),

                        html.Label("Tipo de entrada"),
                        dcc.RadioItems(
                            id="nm-kind",
                            options=[{"label": "Univariado (x → y)", "value": "uni"},
                                     {"label": "Multivariado (Vars → y)", "value": "vars"}],
                            value="uni",
                            className="mt-8"
                        ),

                        html.Div(id="nm-vars-spec-wrap", className="mt-8", style={"display": "none"}, children=[
                            html.Label("Ordem das variáveis (VARS_SPEC)"),
                            dcc.Input(id="nm-vars-spec", type="text",
                                      placeholder="Ex.: IDADE1, IDADE2, DMAX1", className="input"),
                        ]),

                        html.Label("Equação (use nomes de variáveis e β)"),
                        dcc.Textarea(
                            id="nm-expr",
                            placeholder="Ex.: b0*exp(b1/x)  ou  DMAX1*exp(-(b0**((IDADE2**b1)-(IDADE1**b1))))",
                            style={"width": "100%", "height": "100px"},
                            className="mt-8"
                        ),

                        html.Label("Parâmetros (β), separados por vírgula"),
                        dcc.Input(id="nm-params", type="text",
                                  placeholder="Ex.: b0, b1, b2", className="input"),

                        html.Label("Chutes iniciais"),
                        dcc.RadioItems(
                            id="nm-init-kind",
                            options=[{"label": "Automático", "value": "auto"},
                                     {"label": "Manual", "value": "manual"}],
                            value="auto",
                            className="mt-8"
                        ),
                        html.Div(id="nm-init-values-wrap", className="mt-8", style={"display": "none"}, children=[
                            html.Label("Valores iniciais (lista)"),
                            dcc.Input(id="nm-init-values", type="text",
                                      placeholder="Ex.: 0.5, 1.0, 0.1", className="input"),
                        ]),

                        html.Label("Solver"),
                        dcc.Dropdown(
                            id="nm-solver",
                            options=[{"label": "trf", "value": "trf"},
                                     {"label": "dogbox", "value": "dogbox"},
                                     {"label": "lm", "value": "lm"}],
                            value="trf",
                            clearable=False,
                            className="minw-220"
                        ),

                        html.Label("maxfev"),
                        dcc.Input(id="nm-maxfev", type="number", value=20000, className="input"),

                        html.Div(className="row mt-14", children=[
                            html.Button("Validar", id="nm-validate", n_clicks=0, className="btn"),
                            html.Button("Salvar", id="nm-save", n_clicks=0, className="btn btn-primary"),
                        ]),
                        html.Div(id="nm-status", className="mono mt-8"),
                    ]
                ),

                # 4) Mapeamento dinâmico
                html.Div(className="panel mt-14", children=[
                    html.H3("Mapeamento de variáveis", className="panel-title"),
                    html.Small("Dinâmico conforme a equação"),
                    html.Div(id="vars-mapping", className="row mt-8"),
                    html.Div(className="mt-8", children=[
                        html.Small("Target Y (coluna alvo)"),
                        dcc.Dropdown(id="map-target", options=[], value=None, clearable=True, className="minw-260"),
                    ]),
                ]),
            ]),

            # -------- Main (Direita) --------
            html.Div(className="dash-main", children=[
                html.Div(className="panel", children=[  # <- removi o sticky daqui
                    html.H3("Visualização", className="panel-title"),
                    dcc.Loading(type="circle", children=[
                        dcc.Graph(
                            id="scatter-plot",
                            figure={},
                            config={"modeBarButtonsToAdd": ["lasso2d", "select2d"]},
                            className="dash-graph",
                        )
                    ]),
                ]),

                # container flex com os dois paineis lado a lado
                html.Div(
                    className="row mt-14",
                    style={"display": "flex", "gap": "16px", "flexWrap": "wrap"},
                    children=[

                        # painel de eixos
                        html.Div(className="panel flex-1", children=[
                            html.H3("Eixos", className="panel-title"),
                            html.Div(style={"display": "flex", "gap": "16px", "flexWrap": "wrap"}, children=[
                                html.Div(className="minw-220", children=[
                                    html.Small("Eixo X"),
                                    dcc.Dropdown(id="x-dd", options=[], value=None, clearable=False,
                                                 className="minw-220"),
                                ]),
                                html.Div(className="minw-220", children=[
                                    html.Small("Eixo Y"),
                                    dcc.Dropdown(id="y-dd", options=[], value=None, clearable=False,
                                                 className="minw-220"),
                                ]),
                            ]),
                        ]),

                        # painel de ações
                        html.Div(className="panel flex-1", children=[
                            html.H3("Ações", className="panel-title"),
                            html.Div(className="mt-8 row",
                                     style={"display": "flex", "gap": "8px", "flexWrap": "wrap"},
                                     children=[
                                         html.Button("Remover selecionados", id="remove-btn", n_clicks=0,
                                                     className="btn"),
                                         html.Button("Mostrar IDs removidos", id="show-removed-btn", n_clicks=0,
                                                     className="btn"),
                                         html.Button("Resetar base", id="reset-btn", n_clicks=0, className="btn"),
                                         html.Button("Ajustar equação", id="fit-btn", n_clicks=0,
                                                     className="btn btn-primary"),
                                     ]),
                        ]),
                    ]
                ),

                dcc.Loading(type="dot", children=[
                    html.Div(id="status", className="mono mt-14")
                ]),
                html.Div(id="output", className="mono mt-14"),
            ]),
        ]),
    ])
