from flask import Flask, request, render_template
import pandas as pd
import joblib

app = Flask(__name__)

# Load model
model = joblib.load("final_delivery_model.pkl")

@app.route('/')
def index():
    return render_template("dashboard.html")

@app.route('/dashboard')
def dashboard():
    return render_template("dashboard.html", active_page='dashboard')

@app.route('/powerBI')
def powerbi():
    return render_template("powerBI.html", active_page='powerbi')

@app.route('/prediction', methods=["GET", "POST"])
def prediction():
    if request.method == "POST":
        input_data = {
            'Discount_offered': float(request.form['discount']),
            'Weight_in_gms': float(request.form['weight']),
            'Product_importance': request.form['importance'],
            'Category_Cost_of_the_Product': request.form['cost_category'],
            'Prior_purchase_category': request.form['prior_category'],
            'Mode_of_Shipment': request.form['shipment_mode']  # <-- new line
        }

        df = pd.DataFrame([input_data])
        prob = model.predict_proba(df)[0][1]
        prediction = "On Time" if prob >= 0.48 else "Delayed"

        return render_template("prediction.html", prediction=prediction, probability=round(prob*100, 2))
    
    return render_template("prediction.html", active_page='prediction')

if __name__ == '__main__':
    app.run(debug=True)
