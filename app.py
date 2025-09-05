import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.getcwd(), "src")))

from mlproject.pipelines.prediction_pipeline import CustomData, PredictPipeline
from flask import Flask, request, render_template

application = Flask(__name__, template_folder="templates")
app = application

# Root route -> renders home page with form
@app.route('/', methods=['GET'])
def home():
    return render_template('home.html')

# Prediction route -> handles form POST
@app.route('/predictdata', methods=['POST'])
def predict_datapoint():
    try:
        data = CustomData(
            gender=request.form.get('gender'),
            race_ethnicity=request.form.get('ethnicity'),
            parental_level_of_education=request.form.get('parental_level_of_education'),
            lunch=request.form.get('lunch'),
            test_preparation_course=request.form.get('test_preparation_course'),
            reading_score=float(request.form.get('reading_score')),
            writing_score=float(request.form.get('writing_score'))
        )

        pred_df = data.get_data_as_data_frame()
        predict_pipeline = PredictPipeline()
        results = predict_pipeline.predict(pred_df)

        # Render same page with results
        return render_template('home.html', results=results[0])

    except Exception as e:
        return f"Error occurred: {e}"

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
