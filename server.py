from flask import Flask, render_template, request
import numpy as np
import joblib

app = Flask(
    __name__,
    template_folder="webpages",
    static_folder="assets"
)

model = joblib.load("student_predictor.pkl")

@app.route("/")
def home():
    return render_template("landing.html")

@app.route("/student-form")
def student_form():
    return render_template("student_form.html")

@app.route("/prediction", methods=["POST"])
def prediction():

    sex = int(request.form["sex"])
    age = int(request.form["age"])
    schoolsup = int(request.form["schoolsup"])
    famsup = int(request.form["famsup"])
    internet = int(request.form["internet"])
    study = int(request.form["study"])
    failure = int(request.form["failure"])
    absent = int(request.form["absent"])
    g1 = int(request.form["g1"])
    g2 = int(request.form["g2"])

    data = np.array([[
        sex,
        age,
        schoolsup,
        famsup,
        internet,
        study,
        failure,
        absent,
        g1,
        g2
    ]])

    result = model.predict(data)

    answer = (
        "⚠ High Risk of Dropout"
        if result[0] == 1
        else "✅ Low Risk of Dropout"
    )

    return render_template(
        "output.html",
        answer=answer
    )

if __name__ == "__main__":
    app.run(debug=True)