from dash import dcc, html

def build_layout(EQ_FUNCS, VARS_SPEC):
    eq_names = sorted(EQ_FUNCS.keys())

    return html.Div(className="container", children=[
        # Stores base
        dcc.Store(id="orig-data", data=None),
        dcc.Store(id="base-data", data=None),
        dcc.Store(id="removed-ids", data=[]),
        dcc.Store(id="fit-state", data=None),

        # Stores dinâmicos
        dcc.Store(id="store-num-cols", data=[]),
        dcc.Store(id="store-all-cols", data=[]),

        html.H2("Ajustes WB - @ipp", style={"margin": "12px 0 18px"}),

        # 1) Upload
        html.Div(className="panel", children=[
            html.Div(className="row", children=[
                dcc.Upload(
                    id="file-upload",
                    children=html.Button("File"),
                    multiple=False,
                ),
                html.Small("  Selecione um .xlsx ou .csv", style={"marginLeft": "8px", "color": "#666"})
            ])
        ]),

        # 2) Estratificação (container separado)
        html.Div(className="panel mt-14", children=[
            html.Label("Estratificação"),
            dcc.Checklist(
                id="strat-check",
                options=[{"label": " habilitar", "value": "on"}],
                value=[],  # desabilitado por padrão
                style={"marginTop": "6px"}
            ),
            html.Div(className="mt-8", children=[
                html.Small("Coluna de estratos"),
                dcc.Dropdown(id="estrato-col-dd", options=[], value=None, clearable=True, className="minw-220"),
            ]),
            html.Div(className="mt-8", children=[
                html.Small("Valor do estrato"),
                dcc.Dropdown(id="estrato-val-dd", options=[], value=None, clearable=True, className="minw-220"),
            ]),
        ]),

        # 3) Eixos + Equação
        html.Div(className="panel mt-14", children=[
            html.Div(className="row", style={"display": "flex", "gap": "16px", "flexWrap": "wrap"}, children=[
                html.Div(className="minw-220", children=[
                    html.Label("Eixo X"),
                    dcc.Dropdown(id="x-dd", options=[], value=None, clearable=False, className="minw-220"),
                ]),
                html.Div(className="minw-220", children=[
                    html.Label("Eixo Y"),
                    dcc.Dropdown(id="y-dd", options=[], value=None, clearable=False, className="minw-220"),
                ]),
                html.Div(className="minw-260", children=[
                    html.Label("Equação (Eq.py)"),
                    dcc.Dropdown(
                        id="eq-dd",
                        options=[{"label": n, "value": n} for n in eq_names],
                        value=(eq_names[0] if eq_names else None),
                        clearable=True,
                        className="minw-260"
                    ),
                ]),
            ])
        ]),

        # 4) Mapeamento dinâmico
        html.Div(className="panel mt-14", children=[
            html.Label("Mapeamento de variáveis (dinâmico conforme a equação)"),
            html.Div(id="vars-mapping", className="row mt-8"),
            html.Div(className="mt-8", children=[
                html.Small("Target Y (coluna alvo)"),
                dcc.Dropdown(id="map-target", options=[], value=None, clearable=True, className="minw-260"),
            ]),
        ]),

        # 5) Gráfico + Ações
        html.Div(className="panel mt-14", children=[
            dcc.Graph(
                id="scatter-plot",
                figure={},  # vazio até carregar dados
                config={"modeBarButtonsToAdd": ["lasso2d", "select2d"]}
            ),
            html.Div(className="mt-8 row", style={"display": "flex", "gap": "8px", "flexWrap": "wrap"}, children=[
                html.Button("Remover selecionados", id="remove-btn", n_clicks=0),
                html.Button("Mostrar IDs removidos", id="show-removed-btn", n_clicks=0),
                html.Button("Resetar base", id="reset-btn", n_clicks=0),
                html.Button("Ajustar equação", id="fit-btn", n_clicks=0),
            ]),
        ]),

        # Mensagens
        html.Div(id="status", className="mono mt-14"),
        html.Div(id="output", className="mono mt-14"),
    ])
