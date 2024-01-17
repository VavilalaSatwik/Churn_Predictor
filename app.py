from flask import Flask, render_template, request
import numpy as np
import pandas as pd
import pickle

app = Flask(__name__)

with open('model.pkl', 'rb') as model_file:
    classifier = pickle.load(model_file)

# Load the scaler
with open('scaler.pkl', 'rb') as scaler_file:
    scaler = pickle.load(scaler_file)

# Load the expected column names
with open('expected_columns.pkl', 'rb') as ec_file:
    expected_columns = pickle.load(ec_file)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        # Get input from the form
        try:
            wmw = float(request.form['weekly_mins_watched'])
            csc = float(request.form['customer_support_calls'])
            msy = int(request.form['multi_screen_yes'])

            if not isinstance(wmw, (int, float)) or not isinstance(csc, (int, float)):
                raise ValueError("Weekly mins watched and customer support calls must be numeric.")

            if msy == 1:
                msy = 0
            else:
                msy = 1

            # Prepare the input data as a DataFrame
            user_data = pd.DataFrame({
                'weekly_mins_watched': [wmw],
                'customer_support_calls': [csc],
                'multi_screen_yes_1': [msy]
            })

            # Ensure that the column names in user_data match the expected_columns
            user_data = user_data.reindex(columns=expected_columns, fill_value=0)

            # Standardize numerical features using the fitted scaler
            numerical_features = ["weekly_mins_watched", "customer_support_calls"]
            user_data[numerical_features] = scaler.transform(user_data[numerical_features])

            # Use the trained model to make predictions
            prediction = classifier.predict(user_data)

            if (not (0 <= wmw <= 530) or not (0 <= csc <= 10)):
                result_text = "Kindly provide values that are within the acceptable span!"
                result_color = "#1260CC"
                reaction = "https://cdn-icons-gif.flaticon.com/12756/12756664.gif"

                return render_template('result.html', result_text=result_text, result_color=result_color,
                                       reaction=reaction)

            # Display the prediction result
            if prediction == 0:
                result_text = "The customer is predicted not to churn."
                result_color = "#3CD100"
                reaction = "https://cdn-icons-gif.flaticon.com/11175/11175727.gif"
            else:
                result_text = "The customer is predicted to churn."
                result_color = "#EC0202"
                reaction = "https://cdn-icons-gif.flaticon.com/11201/11201849.gif"

            return render_template('result.html', result_text=result_text, result_color=result_color, reaction=reaction)

        except ValueError as e:
            result_text = "Kindly provide appropriate values.!"
            result_color = "#1260CC"
            reaction = "https://cdn-icons-gif.flaticon.com/12756/12756304.gif"
            return render_template('result.html', result_text=result_text, result_color=result_color, reaction=reaction)


#if __name__ == '__main__':
#  app.run(debug=True)
