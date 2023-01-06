import pandas as pd
import io
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.externals import joblib

class CreditCardDefaultPrediction:
    def __init__(self, model_file=None):
        if model_file:
            self.model = joblib.load(model_file)
        else:
            self.model = LogisticRegression()

    def predict(self, input_data):
        # Make predictions using the input data
        predictions = self.model.predict(input_data)
        
        # Add the predictions to the input data as a new column
        input_data['default'] = predictions
        
        # Return the input data with the predictions
        return input_data
    
    def save_model(self, model_file):
        # Save the model to the specified file
        joblib.dump(self.model, model_file)
