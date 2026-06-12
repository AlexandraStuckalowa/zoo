from flask import Flask, render_template, request
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import os

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

app = Flask(__name__)

UPLOAD_FOLDER = "data"
CHARTS_FOLDER = "static/charts"

os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(CHARTS_FOLDER, exist_ok=True)


@app.route("/", methods=["GET", "POST"])
def index():
    if request.method == "GET":
        return render_template("index.html", data_loaded=False)

    uploaded_file = request.files.get("file")

    if uploaded_file is None or uploaded_file.filename == "":
        return render_template(
            "index.html",
            data_loaded=False,
            error="Файл не выбран"
        )

    file_path = os.path.join(UPLOAD_FOLDER, uploaded_file.filename)
    uploaded_file.save(file_path)

    df = pd.read_csv(file_path)

    rows_count = len(df)
    columns_count = len(df.columns)
    average_visitors = round(df["visitors"].mean(), 2)
    max_visitors = int(df["visitors"].max())
    high_days = int(df["high_attendance"].sum())

    weather_names = {
        "sunny": "солнечно",
        "cloudy": "облачно",
        "rainy": "дождь",
        "snowy": "снег"
    }

    weekday_names = {
        "Monday": "понедельник",
        "Tuesday": "вторник",
        "Wednesday": "среда",
        "Thursday": "четверг",
        "Friday": "пятница",
        "Saturday": "суббота",
        "Sunday": "воскресенье"
    }

    attendance_names = {
        0: "обычная",
        1: "высокая"
    }

    weather_counts = df["weather"].value_counts()
    weather_counts.index = weather_counts.index.map(weather_names)

    plt.figure(figsize=(8, 5))
    weather_counts.plot(kind="bar")
    plt.title("Количество дней по типу погоды")
    plt.xlabel("Погода")
    plt.ylabel("Количество дней")
    plt.xticks(rotation=0)
    plt.tight_layout()
    plt.savefig("static/charts/weather_chart.png")
    plt.close()

    weekday_order = [
        "Monday",
        "Tuesday",
        "Wednesday",
        "Thursday",
        "Friday",
        "Saturday",
        "Sunday"
    ]

    weekday_visitors = df.groupby("weekday")["visitors"].mean().reindex(weekday_order)
    weekday_visitors.index = weekday_visitors.index.map(weekday_names)

    plt.figure(figsize=(9, 5))
    weekday_visitors.plot(kind="bar")
    plt.title("Среднее количество посетителей по дням недели")
    plt.xlabel("День недели")
    plt.ylabel("Среднее количество посетителей")
    plt.xticks(rotation=30)
    plt.tight_layout()
    plt.savefig("static/charts/weekday_chart.png")
    plt.close()

    attendance_counts = df["high_attendance"].value_counts().sort_index()
    attendance_counts.index = attendance_counts.index.map(attendance_names)

    plt.figure(figsize=(8, 5))
    attendance_counts.plot(kind="bar")
    plt.title("Распределение обычной и высокой посещаемости")
    plt.xlabel("Тип посещаемости")
    plt.ylabel("Количество дней")
    plt.xticks(rotation=0)
    plt.tight_layout()
    plt.savefig("static/charts/high_attendance_chart.png")
    plt.close()

    features = [
        "weekday",
        "weather",
        "temperature",
        "is_holiday",
        "ticket_price",
        "event"
    ]

    target = "high_attendance"

    X = df[features]
    y = df[target]

    categorical_features = ["weekday", "weather"]
    numeric_features = ["temperature", "is_holiday", "ticket_price", "event"]

    preprocessor = ColumnTransformer(
        transformers=[
            ("cat", OneHotEncoder(handle_unknown="ignore"), categorical_features),
            ("num", "passthrough", numeric_features)
        ]
    )

    model = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("classifier", RandomForestClassifier(random_state=42))
        ]
    )

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42
    )

    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    accuracy = round(accuracy_score(y_test, y_pred), 3)

    example_data = pd.DataFrame([{
        "weekday": "Saturday",
        "weather": "sunny",
        "temperature": 24,
        "is_holiday": 1,
        "ticket_price": 700,
        "event": 1
    }])

    example_prediction = int(model.predict(example_data)[0])

    if example_prediction == 1:
        example_result = "высокая посещаемость"
    else:
        example_result = "обычная посещаемость"

    return render_template(
        "index.html",
        data_loaded=True,
        filename=uploaded_file.filename,
        rows_count=rows_count,
        columns_count=columns_count,
        average_visitors=average_visitors,
        max_visitors=max_visitors,
        high_days=high_days,
        accuracy=accuracy,
        example_result=example_result
    )


@app.route("/reset", methods=["POST"])
def reset():
    for filename in os.listdir(UPLOAD_FOLDER):
        if filename.endswith(".csv"):
            file_path = os.path.join(UPLOAD_FOLDER, filename)
            os.remove(file_path)

    for filename in os.listdir(CHARTS_FOLDER):
        if filename.endswith(".png"):
            file_path = os.path.join(CHARTS_FOLDER, filename)
            os.remove(file_path)

    return render_template("index.html", data_loaded=False)


if __name__ == "__main__":
    app.run(debug=True, use_reloader=False)