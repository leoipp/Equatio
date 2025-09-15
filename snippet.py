from webapp import create_app
from webapp.extensions import db
from webapp.models import User

app = create_app()
with app.app_context():
    email = "admin@exemplo.com"
    u = User.query.filter_by(email=email).first()
    if not u:
        u = User(name="Admin", email=email)
        u.set_password("123456")   # troque em seguida!
        u.email_ok = True
        db.session.add(u); db.session.commit()
        print("Usuário admin criado.")
    else:
        print("Usuário já existe.")