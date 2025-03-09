from flask import Flask, request, render_template
import joblib
import numpy as np

# Load trained model
model = joblib.load("house_price_model.pkl")

# Initialize Flask app
app = Flask(__name__)

@app.route("/", methods=["GET", "POST"])
def index():
    if request.method == "POST":
        try:
            # Get user input from the form
            features = [float(request.form[feature]) for feature in [
                "crim", "zn", "indus", "chas", "nox", "rm", "age",
                "dis", "rad", "tax", "ptratio", "b", "lstat"
            ]]
            
            # Convert input into a NumPy array and reshape it for model prediction
            input_data = np.array(features).reshape(1, -1)

            # Get prediction
            prediction = model.predict(input_data)[0]
            return render_template("index.html", prediction=round(prediction, 2))
        
        except Exception as e:
            return render_template("index.html", error=str(e))
    
    return render_template("index.html", prediction=None, error=None)

if __name__ == "__main__":
    app.run(debug=True)
