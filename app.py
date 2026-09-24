from flask import Flask, render_template, request, redirect, url_for, send_file, session
from flask_sqlalchemy import SQLAlchemy
from werkzeug.security import generate_password_hash, check_password_hash
from functools import wraps
from reportlab.pdfgen import canvas
import pandas as pd
import joblib
import io


app = Flask(__name__)

# =====================================================
# FLASK CONFIGURATION
# =====================================================

app.secret_key = "home-value-ai-secret-key"

app.config["SQLALCHEMY_DATABASE_URI"] = "sqlite:///homevalue.db"
app.config["SQLALCHEMY_TRACK_MODIFICATIONS"] = False

db = SQLAlchemy(app)


# =====================================================
# LOAD MACHINE LEARNING MODEL
# =====================================================

model = joblib.load("model/house_price_model.pkl")


# =====================================================
# USER MODEL
# =====================================================

class User(db.Model):

    id = db.Column(db.Integer, primary_key=True)

    name = db.Column(db.String(100), nullable=False)

    email = db.Column(db.String(120), unique=True, nullable=False)

    password = db.Column(db.String(200), nullable=False)


# =====================================================
# FEEDBACK MODEL
# =====================================================

class Feedback(db.Model):

    id = db.Column(db.Integer, primary_key=True)

    user_id = db.Column(db.Integer, nullable=False)

    message = db.Column(db.Text, nullable=False)

    rating = db.Column(db.Integer, nullable=False)


# =====================================================
# PREDICTION HISTORY MODEL
# =====================================================

class PredictionHistory(db.Model):

    id = db.Column(db.Integer, primary_key=True)

    user_id = db.Column(db.Integer, nullable=False)

    location = db.Column(db.String(200), nullable=False)

    area = db.Column(db.Float, nullable=False)

    rooms = db.Column(db.Integer, nullable=False)

    bedrooms = db.Column(db.Integer, nullable=False)

    bathrooms = db.Column(db.Integer, nullable=False)

    property_type = db.Column(db.String(100), nullable=False)

    predicted_price = db.Column(db.Float, nullable=False)


# =====================================================
# LOGIN REQUIRED
# =====================================================

def login_required(f):

    @wraps(f)
    def decorated_function(*args, **kwargs):

        if "user_id" not in session:

            return redirect(url_for("login"))

        return f(*args, **kwargs)

    return decorated_function


# =====================================================
# ADMIN REQUIRED
# =====================================================

def admin_required(f):

    @wraps(f)
    def decorated_function(*args, **kwargs):

        if "admin_logged_in" not in session:

            return redirect(url_for("admin_login"))

        return f(*args, **kwargs)

    return decorated_function


# =====================================================
# HOME
# =====================================================

@app.route("/")
def home():

    return render_template("index.html")


# =====================================================
# REGISTER
# =====================================================

@app.route("/register", methods=["GET", "POST"])
def register():

    if request.method == "POST":

        name = request.form["name"]

        email = request.form["email"]

        password = request.form["password"]

        confirm_password = request.form["confirm_password"]

        if password != confirm_password:

            return "Passwords do not match!"

        existing_user = User.query.filter_by(
            email=email
        ).first()

        if existing_user:

            return "Email already registered!"

        hashed_password = generate_password_hash(password)

        new_user = User(
            name=name,
            email=email,
            password=hashed_password
        )

        db.session.add(new_user)

        db.session.commit()

        return redirect(url_for("login"))

    return render_template("register.html")


# =====================================================
# USER LOGIN
# =====================================================

@app.route("/login", methods=["GET", "POST"])
def login():

    if request.method == "POST":

        email = request.form["email"]

        password = request.form["password"]

        user = User.query.filter_by(
            email=email
        ).first()

        if user and check_password_hash(
            user.password,
            password
        ):

            session["user_id"] = user.id

            session["user_name"] = user.name

            session["user_email"] = user.email

            return redirect(
                url_for("dashboard")
            )

        return "Invalid email or password!"

    return render_template("login.html")


# =====================================================
# USER LOGOUT
# =====================================================

@app.route("/logout")
def logout():

    session.clear()

    return redirect(
        url_for("login")
    )


# =====================================================
# USER DASHBOARD
# =====================================================

@app.route("/dashboard")
@login_required
def dashboard():

    return render_template(
        "dashboard.html",
        user_name=session["user_name"]
    )


# =====================================================
# HOUSE PRICE PREDICTION
# =====================================================

@app.route("/predict", methods=["GET", "POST"])
@login_required
def predict():

    if request.method == "POST":

        location = request.form["location"]

        area = float(
            request.form["area"]
        )

        rooms = int(
            request.form["rooms"]
        )

        bedrooms = int(
            request.form["bedrooms"]
        )

        bathrooms = int(
            request.form["bathrooms"]
        )

        property_type = request.form[
            "property_type"
        ]

        # MODEL INPUT

        input_data = pd.DataFrame([
            {
                "bhk": bedrooms,
                "propertytype": property_type,
                "location": location,
                "sqft": area
            }
        ])

        # PREDICTION

        prediction = model.predict(
            input_data
        )[0]

        predicted_price = f"₹ {prediction:,.0f}"

        # SAVE PREDICTION HISTORY

        new_prediction = PredictionHistory(

            user_id=session["user_id"],

            location=location,

            area=area,

            rooms=rooms,

            bedrooms=bedrooms,

            bathrooms=bathrooms,

            property_type=property_type,

            predicted_price=float(prediction)

        )

        db.session.add(new_prediction)

        db.session.commit()

        # RESULT PAGE

        return render_template(

            "result.html",

            location=location,

            area=area,

            rooms=rooms,

            bedrooms=bedrooms,

            bathrooms=bathrooms,

            property_type=property_type,

            predicted_price=predicted_price

        )

    return render_template(
        "predict.html"
    )


# =====================================================
# PREDICTION HISTORY
# =====================================================

@app.route("/prediction-history")
@login_required
def prediction_history():

    predictions = PredictionHistory.query.filter_by(

        user_id=session["user_id"]

    ).order_by(

        PredictionHistory.id.desc()

    ).all()

    return render_template(

        "prediction_history.html",

        predictions=predictions

    )


# =====================================================
# DELETE USER PREDICTION
# =====================================================

@app.route(
    "/delete-prediction/<int:prediction_id>",
    methods=["POST"]
)
@login_required
def delete_prediction(prediction_id):

    prediction = PredictionHistory.query.filter_by(

        id=prediction_id,

        user_id=session["user_id"]

    ).first()

    if prediction:

        db.session.delete(prediction)

        db.session.commit()

    return redirect(
        url_for("prediction_history")
    )


# =====================================================
# MARKET ANALYTICS
# =====================================================

@app.route("/analytics")
@login_required
def analytics():

    data = pd.read_csv(
        "dataset/House_Price_Data.csv"
    )

    total_properties = len(data)

    average_price = data[
        "totalprice"
    ].mean()

    average_price_per_sqft = data[
        "pricepersqft"
    ].mean()

    property_counts = data[
        "propertytype"
    ].value_counts()

    property_labels = (
        property_counts.index.tolist()
    )

    property_values = (
        property_counts.values.tolist()
    )

    return render_template(

        "analytics.html",

        total_properties=total_properties,

        average_price=f"₹ {average_price:,.0f}",

        average_price_per_sqft=
        f"₹ {average_price_per_sqft:,.0f}",

        property_labels=property_labels,

        property_values=property_values

    )


# =====================================================
# LOCATION INSIGHTS
# =====================================================

@app.route(
    "/location-insights",
    methods=["GET", "POST"]
)
@login_required
def location_insights():

    data = pd.read_csv(
        "dataset/House_Price_Data.csv"
    )

    locations = sorted(

        data["location"]
        .dropna()
        .unique()
        .tolist()

    )

    selected_location = None

    location_properties = None

    location_average_price = None

    location_average_price_per_sqft = None

    if request.method == "POST":

        selected_location = request.form[
            "location"
        ]

        location_data = data[
            data["location"]
            == selected_location
        ]

        location_properties = len(
            location_data
        )

        location_average_price = (
            location_data["totalprice"]
            .mean()
        )

        location_average_price_per_sqft = (
            location_data["pricepersqft"]
            .mean()
        )

        location_average_price = (
            f"₹ {location_average_price:,.0f}"
        )

        location_average_price_per_sqft = (
            f"₹ {location_average_price_per_sqft:,.0f}"
        )

    return render_template(

        "location_insights.html",

        locations=locations,

        selected_location=selected_location,

        location_properties=location_properties,

        location_average_price=
        location_average_price,

        location_average_price_per_sqft=
        location_average_price_per_sqft

    )


# =====================================================
# PROPERTY COMPARISON
# =====================================================

@app.route(
    "/compare",
    methods=["GET", "POST"]
)
@login_required
def compare():

    if request.method == "POST":

        location1 = request.form[
            "location1"
        ]

        area1 = float(
            request.form["area1"]
        )

        bedrooms1 = int(
            request.form["bedrooms1"]
        )

        property_type1 = request.form[
            "property_type1"
        ]

        location2 = request.form[
            "location2"
        ]

        area2 = float(
            request.form["area2"]
        )

        bedrooms2 = int(
            request.form["bedrooms2"]
        )

        property_type2 = request.form[
            "property_type2"
        ]

        property1_data = pd.DataFrame([

            {
                "bhk": bedrooms1,

                "propertytype": property_type1,

                "location": location1,

                "sqft": area1
            }

        ])

        property2_data = pd.DataFrame([

            {
                "bhk": bedrooms2,

                "propertytype": property_type2,

                "location": location2,

                "sqft": area2
            }

        ])

        prediction1 = model.predict(
            property1_data
        )[0]

        prediction2 = model.predict(
            property2_data
        )[0]

        price1 = (
            f"₹ {prediction1:,.0f}"
        )

        price2 = (
            f"₹ {prediction2:,.0f}"
        )

        return render_template(

            "compare_result.html",

            location1=location1,

            area1=area1,

            bedrooms1=bedrooms1,

            property_type1=property_type1,

            price1=price1,

            location2=location2,

            area2=area2,

            bedrooms2=bedrooms2,

            property_type2=property_type2,

            price2=price2

        )

    return render_template(
        "compare.html"
    )


# =====================================================
# DOWNLOAD PREDICTION REPORT
# =====================================================

@app.route("/download-report")
@login_required
def download_report():

    location = request.args.get(
        "location"
    )

    area = request.args.get(
        "area"
    )

    bedrooms = request.args.get(
        "bedrooms"
    )

    property_type = request.args.get(
        "property_type"
    )

    price = request.args.get(
        "price"
    )

    buffer = io.BytesIO()

    pdf = canvas.Canvas(buffer)

    pdf.setTitle(
        "Home Value AI - Prediction Report"
    )

    pdf.setFont(
        "Helvetica-Bold",
        18
    )

    pdf.drawString(
        50,
        800,
        "Home Value AI - Prediction Report"
    )

    pdf.setFont(
        "Helvetica",
        12
    )

    pdf.drawString(
        50,
        750,
        f"Location: {location}"
    )

    pdf.drawString(
        50,
        725,
        f"Area: {area} sq ft"
    )

    pdf.drawString(
        50,
        700,
        f"Bedrooms: {bedrooms}"
    )

    pdf.drawString(
        50,
        675,
        f"Property Type: {property_type}"
    )

    pdf.drawString(
        50,
        625,
        f"Estimated Price: {price.replace('₹', 'INR')}"
    )

    pdf.drawString(
        50,
        575,
        "Generated by Home Value AI using Machine Learning."
    )

    pdf.save()

    buffer.seek(0)

    return send_file(

        buffer,

        as_attachment=True,

        download_name=
        "Home_Value_AI_Prediction_Report.pdf",

        mimetype="application/pdf"

    )


# =====================================================
# FEEDBACK
# =====================================================

@app.route(
    "/feedback",
    methods=["GET", "POST"]
)
@login_required
def feedback():

    if request.method == "POST":

        message = request.form[
            "message"
        ]

        rating = int(
            request.form["rating"]
        )

        new_feedback = Feedback(

            user_id=session["user_id"],

            message=message,

            rating=rating

        )

        db.session.add(
            new_feedback
        )

        db.session.commit()

        return redirect(
            url_for("dashboard")
        )

    return render_template(
        "feedback.html"
    )


# =====================================================
# ADMIN LOGIN
# =====================================================

@app.route(
    "/admin-login",
    methods=["GET", "POST"]
)
def admin_login():

    if request.method == "POST":

        email = request.form[
            "email"
        ]

        password = request.form[
            "password"
        ]

        admin_email = (
            "namratazalte17@gmail.com"
        )

        admin_password = (
            "namrata2026"
        )

        if (
            email == admin_email
            and password == admin_password
        ):

            session[
                "admin_logged_in"
            ] = True

            session[
                "admin_email"
            ] = email

            return redirect(
                url_for(
                    "admin_dashboard"
                )
            )

        return (
            "Invalid Admin Email or Password!"
        )

    return render_template(
        "admin_login.html"
    )


# =====================================================
# ADMIN DASHBOARD
# =====================================================

@app.route("/admin-dashboard")
@admin_required
def admin_dashboard():

    total_users = User.query.count()

    total_feedback = Feedback.query.count()

    total_predictions = PredictionHistory.query.count()

    return render_template(

        "admin_dashboard.html",

        total_users=total_users,

        total_feedback=total_feedback,

        total_predictions=total_predictions

    )


# =====================================================
# ADMIN USERS
# =====================================================

@app.route("/admin-users")
@admin_required
def admin_users():

    users = User.query.order_by(
        User.id.asc()
    ).all()

    return render_template(

        "admin_users.html",

        users=users

    )


# =====================================================
# ADMIN FEEDBACK
# =====================================================

@app.route("/admin-feedback")
@admin_required
def admin_feedback():

    feedbacks = Feedback.query.order_by(
        Feedback.id.desc()
    ).all()

    return render_template(

        "admin_feedback.html",

        feedbacks=feedbacks

    )


# =====================================================
# ADMIN PREDICTIONS
# =====================================================

@app.route("/admin-predictions")
@admin_required
def admin_predictions():

    predictions = PredictionHistory.query.order_by(
        PredictionHistory.id.desc()
    ).all()

    return render_template(

        "admin_predictions.html",

        predictions=predictions

    )


# =====================================================
# ADMIN LOGOUT
# =====================================================

@app.route("/admin-logout")
def admin_logout():

    session.pop(
        "admin_logged_in",
        None
    )

    session.pop(
        "admin_email",
        None
    )

    return redirect(
        url_for("admin_login")
    )


# =====================================================
# CREATE DATABASE TABLES
# =====================================================

with app.app_context():

    db.create_all()


# =====================================================
# RUN APPLICATION
# =====================================================

if __name__ == "__main__":

    app.run(
        debug=True
    )