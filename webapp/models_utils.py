import json
from sqlalchemy import asc, desc
from flask_login import current_user
from .models import db, UserFitHistory

MAX_FITS_PER_USER = 10

def save_fit_history(user_id:int, fit_info:dict, metrics:dict, x_col=None, y_col=None):
    """
    Salva um ajuste (fit) e garante no máximo 10 por usuário (apaga os mais antigos).
    fit_info: {"name": str, "params": [floats], "uses_vars": bool, "maps": dict|None, "target": str|None}
    metrics:  {"r":float, "r2":float, "rmse":float, "bias":float}
    """
    if not user_id:
        return

    # cria registro
    rec = UserFitHistory(
        user_id=user_id,
        eq_name=fit_info.get("name"),
        params=json.dumps(fit_info.get("params", [])),
        uses_vars=bool(fit_info.get("uses_vars")),
        maps_json=json.dumps(fit_info.get("maps") or {}),
        target_col=fit_info.get("target"),
        r=metrics.get("r"),
        r2=metrics.get("r2"),
        rmse=metrics.get("rmse"),
        bias=metrics.get("bias"),
        x_col=x_col,
        y_col=y_col,
    )
    db.session.add(rec)
    db.session.flush()  # pega ID se precisar

    # aplica política de retenção (mantém os 10 mais novos)
    q = UserFitHistory.query.filter_by(user_id=user_id).order_by(desc(UserFitHistory.created_at))
    ids_keep = [row.id for row in q.limit(MAX_FITS_PER_USER).all()]
    if len(ids_keep) == MAX_FITS_PER_USER:
        # delete o que não está em ids_keep
        UserFitHistory.query.filter(
            UserFitHistory.user_id == user_id,
            ~UserFitHistory.id.in_(ids_keep)
        ).delete(synchronize_session=False)

    db.session.commit()
