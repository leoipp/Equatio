from datetime import datetime
from passlib.hash import bcrypt
from flask_login import UserMixin
from .extensions import db

class User(db.Model, UserMixin):
    __tablename__ = "users"
    id          = db.Column(db.Integer, primary_key=True)
    email       = db.Column(db.String(255), unique=True, nullable=False, index=True)
    name        = db.Column(db.String(120), nullable=False)
    last_name   = db.Column(db.String(120))

    address     = db.Column(db.String(255))
    cpf         = db.Column(db.String(32))

    education   = db.Column(db.String(255))
    profession  = db.Column(db.String(255))
    company     = db.Column(db.String(255))

    password_h  = db.Column(db.String(255), nullable=False)
    is_active   = db.Column(db.Boolean, default=True)
    email_ok    = db.Column(db.Boolean, default=False)
    created_at  = db.Column(db.DateTime, default=datetime.utcnow)

    # login social
    oauth_provider = db.Column(db.String(50), index=True)  # "google" | "facebook" | None
    oauth_sub      = db.Column(db.String(255), index=True) # ID no provedor

    def set_password(self, raw):
        self.password_h = bcrypt.hash(raw)

    def check_password(self, raw):
        try:
            return bcrypt.verify(raw, self.password_h)
        except Exception:
            return False


class UserModel(db.Model):
    __tablename__ = "user_models"
    id         = db.Column(db.Integer, primary_key=True)
    user_id    = db.Column(db.Integer, db.ForeignKey("users.id"), index=True, nullable=False)

    name       = db.Column(db.String(128), nullable=False)   # único POR usuário
    kind       = db.Column(db.String(10), nullable=False)    # "uni" | "vars"
    expr       = db.Column(db.Text, nullable=False)          # expressão SymPy
    params     = db.Column(db.Text, nullable=False)          # "b0,b1,b2"
    vars_list  = db.Column(db.Text)                          # "IDADE1,IDADE2,DMAX1" (se vars)
    init_kind  = db.Column(db.String(10), default="auto")    # "auto" | "manual"
    init_values= db.Column(db.Text)                          # "0.5,1.0" (se manual)
    solver     = db.Column(db.String(10), default="trf")     # "trf" | "dogbox" | "lm"
    maxfev     = db.Column(db.Integer, default=20000)

    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    updated_at = db.Column(db.DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

    __table_args__ = (
        db.UniqueConstraint("user_id", "name", name="uq_user_model_per_user"),
    )


# Histórico de ajustes (limite de 10 por usuário)
class UserFitHistory(db.Model):
    __tablename__ = "user_fit_history"
    id         = db.Column(db.Integer, primary_key=True)
    user_id    = db.Column(db.Integer, db.ForeignKey("users.id"), index=True, nullable=False)

    eq_name    = db.Column(db.String(128), nullable=False)   # qual modelo foi ajustado
    params     = db.Column(db.Text, nullable=False)          # JSON string dos parâmetros
    uses_vars  = db.Column(db.Boolean, default=False)
    maps_json  = db.Column(db.Text)                          # JSON dos mapeamentos (se vars)
    target_col = db.Column(db.String(128))                   # coluna alvo (se vars)

    # métricas
    r          = db.Column(db.Float)
    r2         = db.Column(db.Float)
    rmse       = db.Column(db.Float)
    bias       = db.Column(db.Float)

    # opcional: quais colunas X/Y foram usadas no gráfico
    x_col      = db.Column(db.String(128))
    y_col      = db.Column(db.String(128))

    created_at = db.Column(db.DateTime, default=datetime.utcnow, index=True)
