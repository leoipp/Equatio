import importlib
from dash import Dash
from .app_layout import build_layout
from .callbacks import register_callbacks

def init_dash(server):
    """
    Monta o Dash como 'filho' do Flask em /dash
    """
    dash_app = Dash(
        __name__,
        server=server,
        url_base_pathname="/dash/",
        suppress_callback_exceptions=True,
        title="Ajustes WB - @ipp",
    )

    # Carrega Eq.py (local ao pacote dashapp)
    Eq = importlib.import_module("webapp.dashapp.Eq")
    importlib.reload(Eq)

    EQ_FUNCS   = Eq.EQUATIONS
    VARS_SPEC  = getattr(Eq, "VARS_SPEC", {})
    # Layout
    dash_app.layout = build_layout(EQ_FUNCS, VARS_SPEC)
    # Callbacks
    register_callbacks(dash_app, context=dict(EQ_FUNCS=EQ_FUNCS, VARS_SPEC=VARS_SPEC))

    return dash_app
