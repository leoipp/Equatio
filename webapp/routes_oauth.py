# webapp/routes_oauth.py
import os
from flask import Blueprint, url_for, redirect, session, flash
from flask_login import login_user
from webapp.oauth import oauth
from webapp.models import db, User

bp_oauth = Blueprint("oauth", __name__)

def _redirect_uri(provider):
    base = os.getenv("OAUTH_REDIRECT_BASE", "http://127.0.0.1:5000")
    return f"{base}{url_for('oauth.callback', provider=provider)}"

@bp_oauth.route("/login/<provider>")
def login(provider):
    if provider == "google":
        return oauth.google.authorize_redirect(_redirect_uri("google"))
    elif provider == "facebook":
        return oauth.facebook.authorize_redirect(_redirect_uri("facebook"))
    else:
        flash("Provedor inválido.", "error")
        return redirect(url_for("login"))

@bp_oauth.route("/auth/callback/<provider>")
def callback(provider):
    if provider == "google":
        token = oauth.google.authorize_access_token()
        userinfo = oauth.google.parse_id_token(token)
        # userinfo: {sub, email, name, picture, ...}
        sub = userinfo.get("sub")
        email = userinfo.get("email")
        name = userinfo.get("name") or email.split("@")[0]

    elif provider == "facebook":
        token = oauth.facebook.authorize_access_token()
        # buscar dados básicos
        resp = oauth.facebook.get("me?fields=id,name,email", token=token)
        data = resp.json()
        sub = data.get("id")
        email = data.get("email")
        name = data.get("name") or (email.split("@")[0] if email else f"fb_{sub}")

    else:
        flash("Provedor inválido.", "error")
        return redirect(url_for("login"))

    if not email:
        flash("Não foi possível obter e-mail do provedor.", "error")
        return redirect(url_for("login"))

    # Vincula usuário: por (provider, sub) OU por email
    user = User.query.filter_by(oauth_provider=provider, oauth_sub=sub).first()
    if not user:
        user = User.query.filter_by(email=email).first()
        if user:
            # vincula conta social a um usuário existente
            user.oauth_provider = provider
            user.oauth_sub = sub
        else:
            # cria novo usuário
            user = User(
                name=name,
                email=email,
                oauth_provider=provider,
                oauth_sub=sub,
            )
            db.session.add(user)
        db.session.commit()

    login_user(user)
    flash("Login realizado com sucesso!", "success")
    return redirect(url_for("home"))
