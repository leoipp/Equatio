import os
import re
from flask import Flask, render_template, request, redirect, url_for, session, flash, g
from flask_babel import Babel, _
from flask_login import LoginManager, login_user, logout_user, current_user, login_required
from itsdangerous import URLSafeTimedSerializer, BadSignature, SignatureExpired
from email_validator import validate_email, EmailNotValidError

from .extensions import db, migrate
from .models import User
from .dashapp import init_dash
from .oauth import init_oauth
from .routes_oauth import bp_oauth
from .routes_billing import bp_billing

def create_app():
    app = Flask(__name__, template_folder="templates", static_folder="static", static_url_path="/static")
    app.config["SECRET_KEY"] = os.getenv("SECRET_KEY", "troque-isto-em-producao")
    app.config["SQLALCHEMY_DATABASE_URI"] = os.getenv("DATABASE_URL", "sqlite:///app.db")
    app.config["SQLALCHEMY_TRACK_MODIFICATIONS"] = False

    # i18n
    app.config['BABEL_DEFAULT_LOCALE'] = 'pt'
    app.config['BABEL_TRANSLATION_DIRECTORIES'] = 'webapp/translations'
    app.config['LANGUAGES'] = ['pt', 'en', 'es']

    # init extensions
    db.init_app(app)
    migrate.init_app(app, db)

    # ---- Babel (forma compatível com 3.x) ----
    babel = Babel()
    def _locale_selector():
        # 1) preferência do usuário (sessão)
        lang = session.get('lang')
        if lang in app.config['LANGUAGES']:
            g.current_lang = lang
            return lang
        # 2) melhor escolha do browser
        best = request.accept_languages.best_match(app.config['LANGUAGES'])
        g.current_lang = best or 'pt'
        return g.current_lang
    babel.init_app(app, locale_selector=_locale_selector)

    # OAuth / Billing
    init_oauth(app)
    app.register_blueprint(bp_oauth)
    app.register_blueprint(bp_billing)

    # login manager
    login_manager = LoginManager()
    login_manager.login_view = "login"
    login_manager.init_app(app)

    @login_manager.user_loader
    def load_user(user_id):
        return db.session.get(User, int(user_id))

    # serializer para verificação de email
    def get_serializer():
        return URLSafeTimedSerializer(app.config["SECRET_KEY"], salt="email-verify")

    # ---- páginas HTML ----
    @app.route("/")
    def home():
        return render_template("home.html")

    @app.route("/login", methods=["GET", "POST"])
    def login():
        if request.method == "POST":
            email = request.form.get("email", "").strip().lower()
            password = request.form.get("password", "")

            user = User.query.filter_by(email=email).first()
            if not user or not user.check_password(password):
                flash(_("E-mail ou senha inválidos."), "error")
                return redirect(url_for("login"))

            login_user(user)
            next_url = request.args.get("next")
            return redirect(next_url or url_for("home"))

        return render_template("login.html")

    @app.route("/logout")
    @login_required
    def logout():
        logout_user()
        return redirect(url_for("home"))

    @app.route("/register", methods=["GET", "POST"])
    def register():
        if request.method == "POST":
            name = request.form.get("name", "").strip()
            email = request.form.get("email", "").strip().lower()
            password = request.form.get("password", "")
            confirm = request.form.get("confirm_password", "")

            if not name or not email or not password or not confirm:
                flash(_("Preencha todos os campos."), "error")
                return redirect(url_for("register"))

            # e-mail válido (já que você importou o validador)
            try:
                validate_email(email)
            except EmailNotValidError as e:
                flash(_("E-mail inválido.") + f" {e}", "error")
                return redirect(url_for("register"))

            if password != confirm:
                flash(_("As senhas não coincidem."), "error")
                return redirect(url_for("register"))

            if User.query.filter_by(email=email).first():
                flash(_("Este e-mail já está cadastrado. Faça login."), "error")
                return redirect(url_for("register"))

            if not re.search(r"[A-Za-z]", password) or not re.search(r"\d", password):
                flash(_("A senha deve conter letras e números."), "error")
                return redirect(url_for("register"))
            if len(password) < 6:
                flash(_("A senha deve ter pelo menos 6 caracteres."), "error")
                return redirect(url_for("register"))

            user = User(name=name, email=email)
            user.set_password(password)
            db.session.add(user)
            db.session.commit()

            flash(_("Conta criada com sucesso! Faça login."), "success")
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
                flash(_("E-mail verificado com sucesso!"), "success")
            else:
                flash(_("Token inválido."), "error")
        except SignatureExpired:
            flash(_("Link expirado. Faça login e solicite novo link."), "error")
        except BadSignature:
            flash(_("Token inválido."), "error")
        return redirect(url_for("login"))

    # Proteger /dash* (e a API do Dash) quando não logado
    @app.before_request
    def protect_dash():
        path = (request.path or "")
        if path.startswith("/dash"):
            if not current_user.is_authenticated:
                return redirect(url_for("login", next=path))
            # Se quiser exigir e-mail verificado:
            # if not current_user.email_ok:
            #     flash(_("Verifique seu e-mail para acessar o Dashboard."), "error")
            #     return redirect(url_for("home"))

    # monta o Dash
    init_dash(app)

    @app.route("/dash-page")
    def dash_page():
        return render_template("dash_wrapper.html")

    @app.route("/profile", methods=["GET", "POST"])
    @login_required
    def profile():
        if request.method == "POST":
            section = request.form.get("section")
            try:
                if section == "pessoais":
                    current_user.name = request.form.get("first_name", "").strip()
                    current_user.last_name = request.form.get("last_name", "").strip()
                    current_user.address = request.form.get("address", "").strip()
                    current_user.cpf = request.form.get("cpf", "").strip()
                elif section == "academico":
                    current_user.education = request.form.get("education", "").strip()
                    current_user.profession = request.form.get("profession", "").strip()
                    current_user.company = request.form.get("company", "").strip()

                db.session.commit()
                flash(_("Informações atualizadas com sucesso."), "success")
            except Exception as e:
                db.session.rollback()
                flash(_("Erro ao salvar: ") + str(e), "error")

            return redirect(url_for("profile"))

        return render_template("profile.html", user=current_user)

    # ---- trocar idioma (POST do select no header) ----
    @app.post('/set-language')
    def set_language():
        lang = request.form.get('lang', 'pt')
        if lang not in app.config['LANGUAGES']:
            lang = 'pt'
        session['lang'] = lang
        # volta para a página de origem; se não houver, vai p/ home
        return redirect(request.referrer or url_for('home'))

    # Expor idiomas no template (opcional)
    @app.context_processor
    def inject_langs():
        return dict(LANGUAGES=app.config['LANGUAGES'])

    return app
