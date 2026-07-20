from flask import Flask, render_template, request
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import math

app = Flask(__name__)


model = None
X = None
y = None
y_pred = None

prediction = None
mae = None
mse = None
rmse = None
r2 = None


# Convert input data
def convert_data(data):
    return [float(i.strip()) for i in data.split(",")]


# Home page
@app.route("/")
def home():

    return render_template(
        "index.html",
        prediction=None,
        mae=None,
        mse=None,
        rmse=None,
        r2=None,
        graph=False
    )


# ---------------------------------
# Predict Final Marks
# ---------------------------------
@app.route("/predict", methods=["POST"])
def predict():

    global model, X, y, y_pred
    global prediction, mae, mse, rmse, r2

    try:

        # Training dataset

        study = convert_data(request.form["study_hours"])
        attendance = convert_data(request.form["attendance"])
        assignment = convert_data(request.form["assignment"])
        final_marks = convert_data(request.form["final_marks"])


        # Creating input features

        X = np.column_stack(
            (study, attendance, assignment)
        )

        y = np.array(final_marks)


        # Train model internally

        model = LinearRegression()
        model.fit(X, y)


        # Training prediction

        y_pred = model.predict(X)


        # Metrics

        mae = mean_absolute_error(y, y_pred)

        mse = mean_squared_error(y, y_pred)

        rmse = math.sqrt(mse)

        r2 = r2_score(y, y_pred)



        # Testing data

        test_study = float(request.form["test_study"])

        test_attendance = float(request.form["test_attendance"])

        test_assignment = float(request.form["test_assignment"])



        test_data = np.array(
            [[
                test_study,
                test_attendance,
                test_assignment
            ]]
        )


        # Final prediction

        prediction = model.predict(test_data)[0]



        return render_template(
            "index.html",
            prediction=round(float(prediction),2),
            mae=round(float(mae),3),
            mse=round(float(mse),3),
            rmse=round(float(rmse),3),
            r2=round(float(r2),3),
            graph=False
        )


    except Exception as e:

        return "Error : " + str(e)

# Visualization

@app.route("/visualize", methods=["POST"])
def visualize():

    global X,y,y_pred

    global prediction,mae,mse,rmse,r2


    try:

        if X is None:

            return "Please click Predict Final Marks first"


        # Actual vs Predicted graph

        plt.figure(figsize=(7,5))

# Actual points
        plt.scatter(
    range(len(y)),
    y,
    label="Actual Marks"
)

# Predicted points
        plt.scatter(
    range(len(y_pred)),
    y_pred,
    label="Predicted Marks"
)

# Straight line connecting predicted values
        plt.plot(
        range(len(y_pred)),
        y_pred
          )


        plt.title("Actual vs Predicted Final Marks")

        plt.xlabel("Student")

        plt.ylabel("Marks")

        plt.legend()

        plt.grid(True)


        plt.tight_layout()

        plt.savefig("static/prediction.png")

        plt.close()


        # Metrics graph

        plt.figure(figsize=(7,5))


        metric_name=[
            "MAE",
            "MSE",
            "RMSE",
            "R2"
        ]


        metric_value=[
            mae,
            mse,
            rmse,
            r2
        ]


        plt.scatter(
            metric_name,
            metric_value,
            s=120
        )


        for i in range(len(metric_name)):

            plt.text(
                metric_name[i],
                metric_value[i],
                round(metric_value[i],2)
            )


        plt.title(
            "Performance Metrics"
        )

        plt.xlabel(
            "Metrics"
        )

        plt.ylabel(
            "Value"
        )

        plt.grid(True)


        plt.savefig(
            "static/metrics.png"
        )

        plt.close()



        return render_template(
            "index.html",
            prediction=round(float(prediction),2),
            mae=round(float(mae),3),
            mse=round(float(mse),3),
            rmse=round(float(rmse),3),
            r2=round(float(r2),3),
            graph=True
        )


    except Exception as e:

        return "Error : " + str(e)



# Reset

@app.route("/reset")
def reset():

    global model,X,y,y_pred

    global prediction,mae,mse,rmse,r2


    model=None
    X=None
    y=None
    y_pred=None

    prediction=None

    mae=None
    mse=None
    rmse=None
    r2=None


    return render_template(
        "index.html",
        prediction=None,
        mae=None,
        mse=None,
        rmse=None,
        r2=None,
        graph=False
    )



# Run Flask

if __name__=="__main__":

    app.run(debug=True)