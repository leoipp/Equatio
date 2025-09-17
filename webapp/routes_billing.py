# webapp/routes_billing.py
from flask import Blueprint, request, jsonify, url_for
from flask_login import login_required, current_user
import stripe, os

bp_billing = Blueprint("billing", __name__)
stripe.api_key = os.getenv("STRIPE_SECRET_KEY")

PRICE_MAP = {  # defina seus preços no Stripe e mapeie aqui
    "50":  os.getenv("STRIPE_PRICE_50"),
    "200": os.getenv("STRIPE_PRICE_200"),
    "500": os.getenv("STRIPE_PRICE_500"),
    "1000":os.getenv("STRIPE_PRICE_1000"),
}

@bp_billing.route("/billing/checkout", methods=["POST"])
@login_required
def billing_create_checkout():
    data = request.get_json() or {}
    pack = str(data.get("pack", "")).strip()
    price_id = PRICE_MAP.get(pack)
    if not price_id:
        return jsonify({"error": "Pacote inválido."}), 400

    session = stripe.checkout.Session.create(
        mode="payment",
        line_items=[{"price": price_id, "quantity": 1}],
        success_url=url_for("billing.checkout_success", _external=True) + "?session_id={CHECKOUT_SESSION_ID}",
        cancel_url=url_for("billing.checkout_cancel", _external=True),
        client_reference_id=str(current_user.id),
        metadata={"user_id": current_user.id, "pack": pack},
    )
    return jsonify({"url": session.url})

@bp_billing.route("/billing/success")
def checkout_success():
    return "Pagamento aprovado. Você pode fechar esta aba."

@bp_billing.route("/billing/cancel")
def checkout_cancel():
    return "Pagamento cancelado."

@bp_billing.route("/billing/webhook", methods=["POST"])
def stripe_webhook():
    payload = request.get_data(as_text=True)
    sig = request.headers.get("Stripe-Signature", "")
    endpoint_secret = os.getenv("STRIPE_WEBHOOK_SECRET")
    try:
        event = stripe.Webhook.construct_event(payload, sig, endpoint_secret)
    except Exception:
        return "", 400

    if event["type"] == "checkout.session.completed":
        s = event["data"]["object"]
        user_id = int(s.get("client_reference_id"))
        pack = int((s.get("metadata") or {}).get("pack", 0))
        # 1) incrementa créditos do usuário
        from .models import db, User
        u = User.query.get(user_id)
        if u:
            u.credits = (u.credits or 0) + pack
            db.session.commit()
        # 2) salva TopUp (opcional)
        # 3) limite de 10 ajustes por usuário ⇒ trate quando salvar ajustes (ORDER BY created_at DESC LIMIT 10)
    return "", 200
