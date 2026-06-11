import os
from flask import Flask, render_template, url_for, flash, redirect
from flask_sqlalchemy import SQLAlchemy
from forms import RegistrationForm
from models import db, User
from pipeline import get_pipeline_results, build_3d_chart

app = Flask(__name__)
app.config['SECRET_KEY'] = os.environ.get('SECRET_KEY', 'sayza_dev_key_change_in_prod')
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///site.db'
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False

db.init_app(app)

# Pipeline startup'ta bir kez çalışır, sonuçlar cache'lenir
_pipeline = get_pipeline_results()


@app.route("/")
@app.route("/home")
def index():
    rfm = _pipeline["rfm"]
    stats = _pipeline["stats"]

    grafik = build_3d_chart(rfm)
    segment_list = stats.reset_index().to_dict("records")

    return render_template(
        "index.html",
        toplam_musteri=len(rfm),
        n_clusters=rfm["Cluster"].nunique(),
        grafik=grafik,
        segment_list=segment_list,
    )


@app.route("/register", methods=["GET", "POST"])
def register():
    form = RegistrationForm()
    if form.validate_on_submit():
        flash(f"{form.username.data} için hesap oluşturuldu!", "success")
        return redirect(url_for("index"))
    return render_template("register.html", title="Kayıt Ol", form=form)


if __name__ == "__main__":
    with app.app_context():
        db.create_all()
    app.run(debug=True)
