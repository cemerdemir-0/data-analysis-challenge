import os
from flask import Flask, render_template, url_for, flash, redirect, abort, request
from flask_sqlalchemy import SQLAlchemy
from flask_login import LoginManager, login_user, logout_user, login_required, current_user
from werkzeug.security import generate_password_hash, check_password_hash
from forms import RegistrationForm, LoginForm
from models import db, User
from pipeline import get_pipeline_results, build_3d_chart, build_landing_chart
from recommendation import get_customer_recommendations, SEGMENT_INSIGHTS
from time_series import (build_revenue_chart, build_customers_chart,
                         build_segment_area_chart, build_aov_chart, get_key_insights)

app = Flask(__name__)
app.config['SECRET_KEY'] = os.environ.get('SECRET_KEY', 'sayza_dev_key_change_in_prod')
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///site.db'
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False

db.init_app(app)

login_manager = LoginManager(app)
login_manager.login_view = 'login'
login_manager.login_message = 'Bu sayfayı görüntülemek için giriş yapmanız gerekiyor.'
login_manager.login_message_category = 'info'


@login_manager.user_loader
def load_user(user_id):
    return User.query.get(int(user_id))


print("[App] Pipeline yükleniyor...")
_pipeline = get_pipeline_results()
print("[App] Hazır.")


# ── Public routes ──────────────────────────────────────────

@app.route("/")
def landing():
    grafik = build_landing_chart(_pipeline["rfm"])
    segment_list = _pipeline["stats"].reset_index().to_dict("records")
    return render_template("landing.html", grafik=grafik, segment_list=segment_list)


@app.route("/login", methods=["GET", "POST"])
def login():
    if current_user.is_authenticated:
        return redirect(url_for("dashboard"))
    form = LoginForm()
    if form.validate_on_submit():
        user = User.query.filter_by(email=form.email.data).first()
        if user and check_password_hash(user.password, form.password.data):
            login_user(user, remember=form.remember.data)
            next_page = request.args.get("next")
            return redirect(next_page or url_for("dashboard"))
        flash("E-posta veya şifre hatalı.", "danger")
    return render_template("login.html", form=form)


@app.route("/register", methods=["GET", "POST"])
def register():
    if current_user.is_authenticated:
        return redirect(url_for("dashboard"))
    form = RegistrationForm()
    if form.validate_on_submit():
        existing = User.query.filter_by(email=form.email.data).first()
        if existing:
            flash("Bu e-posta zaten kayıtlı.", "danger")
            return render_template("register.html", form=form)
        hashed_pw = generate_password_hash(form.password.data)
        user = User(username=form.username.data, email=form.email.data, password=hashed_pw)
        db.session.add(user)
        db.session.commit()
        flash("Hesabın oluşturuldu! Giriş yapabilirsin.", "success")
        return redirect(url_for("login"))
    return render_template("register.html", title="Kayıt Ol", form=form)


@app.route("/logout")
@login_required
def logout():
    logout_user()
    return redirect(url_for("landing"))


# ── Protected routes ───────────────────────────────────────

@app.route("/dashboard")
@login_required
def dashboard():
    rfm = _pipeline["rfm"]
    stats = _pipeline["stats"]
    segment_insights = _pipeline["segment_insights"]

    grafik = build_3d_chart(rfm)
    segment_list = stats.reset_index().to_dict("records")

    return render_template(
        "index.html",
        toplam_musteri=len(rfm),
        n_clusters=rfm["Cluster"].nunique(),
        grafik=grafik,
        segment_list=segment_list,
        segment_insights=segment_insights,
    )


@app.route("/time-series")
@login_required
def time_series():
    ts = _pipeline["ts_data"]
    summary = ts["summary"]
    segment_monthly = ts["segment_monthly"]

    return render_template(
        "time_series.html",
        chart_revenue=build_revenue_chart(summary),
        chart_customers=build_customers_chart(summary),
        chart_segments=build_segment_area_chart(segment_monthly),
        chart_aov=build_aov_chart(summary),
        insights=get_key_insights(summary),
    )


@app.route("/customer/<int:customer_id>")
@login_required
def customer_detail(customer_id):
    rfm_full = _pipeline["rfm_full"]
    raw_df = _pipeline["raw_df"]

    customer_row, recommendations, segment_avg = get_customer_recommendations(
        customer_id, raw_df, rfm_full
    )
    if customer_row is None:
        abort(404)

    segment_meta = SEGMENT_INSIGHTS.get(customer_row["Segment"], {})

    return render_template(
        "customer.html",
        customer_id=customer_id,
        customer=customer_row,
        recommendations=recommendations.to_dict("records") if recommendations is not None else [],
        segment_avg=segment_avg.to_dict() if segment_avg is not None else {},
        segment_meta=segment_meta,
    )


@app.errorhandler(404)
def not_found(e):
    return render_template("404.html"), 404


if __name__ == "__main__":
    with app.app_context():
        db.create_all()
    app.run(debug=True)
