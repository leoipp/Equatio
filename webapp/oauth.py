# webapp/oauth.py
import os
from authlib.integrations.flask_client import OAuth

oauth = OAuth()

def init_oauth(app):
    oauth.init_app(app)

    oauth.register(
        name="google",
        client_id=os.getenv("GOOGLE_CLIENT_ID"),
        client_secret=os.getenv("GOOGLE_CLIENT_SECRET"),
        server_metadata_url="https://accounts.google.com/.well-known/openid-configuration",
        client_kwargs={"scope": "openid email profile"},
    )

    oauth.register(
        name="facebook",
        client_id=os.getenv("FACEBOOK_CLIENT_ID"),
        client_secret=os.getenv("FACEBOOK_CLIENT_SECRET"),
        access_token_url="https://graph.facebook.com/v18.0/oauth/access_token",
        authorize_url="https://www.facebook.com/v18.0/dialog/oauth",
        api_base_url="https://graph.facebook.com/v18.0/",
        client_kwargs={"scope": "email"},
    )
