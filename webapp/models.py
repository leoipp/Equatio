from datetime import datetime
from passlib.hash import bcrypt
from flask_login import UserMixin
from .extensions import db

class User(db.Model, UserMixin):
    __tablename__ = "users"
    id          = db.Column(db.Integer, primary_key=True)
    email       = db.Column(db.String(255), unique=True, nullable=False, index=True)
    name        = db.Column(db.String(120), nullable=False)
    password_h  = db.Column(db.String(255), nullable=False)
    is_active   = db.Column(db.Boolean, default=True)
    email_ok    = db.Column(db.Boolean, default=False)   # verificação de email
    created_at  = db.Column(db.DateTime, default=datetime.utcnow)

    def set_password(self, raw):
        self.password_h = bcrypt.hash(raw)

    def check_password(self, raw):
        try:
            return bcrypt.verify(raw, self.password_h)
        except Exception:
            return False
