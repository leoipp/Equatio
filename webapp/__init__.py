import os
from flask import Flask, render_template, request, redirect, url_for, session, flash
from flask_login import LoginManager, login_user, logout_user, current_user, login_required
from itsdangerous import URLSafeTimedSerializer, BadSignature, SignatureExpired
from email_validator import validate_email, EmailNotValidError

from .extensions import db, migrate
from .models import User
from .dashapp import init_dash

def create_app():
    app = Flask(__name__, template_folder="templates", static_folder="static", static_url_path="/static")
    app.config["SECRET_KEY"] = os.getenv("SECRET_KEY", "troque-isto-em-producao")
    # DB: use SQLite local ou Postgres em prod (ex.: postgres://user:pass@host/db)
    app.config["SQLALCHEMY_DATABASE_URI"] = os.getenv("DATABASE_URL", "sqlite:///app.db")
    app.config["SQLALCHEMY_TRACK_MODIFICATIONS"] = False

    # init extensions
    db.init_app(app)
    migrate.init_app(app, db)

    # login manager
    login_manager = LoginManager()
    login_manager.login_view = "login"
    login_manager.init_app(app)

    @login_manager.user_loader
    def load_user(user_id):
        return db.session.get(User, int(user_id))

    # token serializer p/ email verification
    def get_serializer():
        return URLSafeTimedSerializer(app.config["SECRET_KEY"], salt="email-verify")

    # ---- páginas HTML ----
    @app.route("/")
    def home():
        return render_template("home.html")

    @app.route("/login", methods=["GET", "POST"])
    def login():
        if request.method == "POST":
            email = request.form.get("email","").strip().lower()
            password = request.form.get("password","")
            user = User.query.filter_by(email=email).first()
            if user and user.check_password(password):
                login_user(user)
                next_url = request.args.get("next") or url_for("home")
                return redirect(next_url)
            flash("Credenciais inválidas.", "error")
        return render_template("login.html")

    @app.route("/logout")
    @login_required
    def logout():
        logout_user()
        return redirect(url_for("home"))

    @app.route("/register", methods=["GET", "POST"])
    def register():
        if request.method == "POST":
            name = request.form.get("name")
            email = request.form.get("email").lower().strip()
            password = request.form.get("password")

            # Verifica email duplicado
            existing = User.query.filter_by(email=email).first()
            if existing:
                flash("Este email já está registrado. Tente fazer login.", "error")
                return redirect(url_for("register"))

            # Validação básica de senha
            if len(password) < 6:
                flash("A senha deve ter pelo menos 6 caracteres.", "error")
                return redirect(url_for("register"))
            if password.isnumeric():
                flash("A senha não pode ser apenas números.", "error")
                return redirect(url_for("register"))

            # Criar usuário
            u = User(name=name, email=email)
            u.set_password(password)
            db.session.add(u)
            db.session.commit()

            flash("Conta criada com sucesso! Faça login.", "success")
            return redirect(url_for("login"))

        return render_template("register.html")

    @app.route("/verify/<token>")
    def verify_email(token):
        s = get_serializer()
        try:
            data = s.loads(token, max_age=60*60*24*3)  # 3 dias
            user = db.session.get(User, int(data["uid"]))
            if user and user.email == data["email"]:
                user.email_ok = True
                db.session.commit()
                flash("E-mail verificado com sucesso!", "success")
            else:
                flash("Token inválido.", "error")
        except SignatureExpired:
            flash("Link expirado. Faça login e solicite novo link.", "error")
        except BadSignature:
            flash("Token inválido.", "error")
        return redirect(url_for("login"))

    # Protege /dash* com login
    @app.before_request
    def protect_dash():
        path = (request.path or "")
        if path.startswith("/dash"):
            if not current_user.is_authenticated:
                return redirect(url_for("login", next=path))
            # opcional: exigir e-mail verificado
            # if not current_user.email_ok:
            #     flash("Verifique seu e-mail para acessar o Dash.", "error")
            #     return redirect(url_for("home"))

    # monta o Dash
    init_dash(app)

    # opcional: página com iframe
    @app.route("/dash-page")
    def dash_page():
        return render_template("dash_wrapper.html")

    return app
