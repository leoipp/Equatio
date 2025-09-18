import importlib
from dash import Dash
from .app_layout import build_layout
from .callbacks import register_callbacks
import logging

def init_dash(server):
    """
    Monta o Dash como 'filho' do Flask em /dash
    """
    dash_app = Dash(
        __name__,
        server=server,
        url_base_pathname="/dash/",
        external_stylesheets=["/static/styles.css"],
        suppress_callback_exceptions=True
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

    # Dash DevTools (útil no dev)
    dash_app.enable_dev_tools(
        debug=True,
        dev_tools_ui=True,
        dev_tools_hot_reload=True,
        dev_tools_props_check=True,
        dev_tools_silence_routes_logging=False,
    )

    # Logging verboso
    logging.basicConfig(level=logging.DEBUG)  # root
    dash_app.logger.setLevel(logging.DEBUG)
    logging.getLogger("werkzeug").setLevel(logging.DEBUG)

    return dash_app
