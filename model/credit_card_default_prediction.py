import pandas as pd
import numpy as np
from sklearn import feature_extraction, metrics
import io
from tqdm import tqdm

class CreditCardDefaultPrediction():
    def __init__(self, dtf_input):
      self.dtf_input = dtf_input


    @staticmethod
    def predict(dtf_input):
        ## fit model
        model = RandomForestClassifier()
        model.fit(dtf_input.iloc[:, :-1], dtf_input.iloc[:, -1])

        ## make predictions
        predictions = model.predict(dtf_input.iloc[:, :-1])
        dtf_predictions = pd.DataFrame(predictions, columns=["default"])
        dtf_predictions["default"] = dtf_predictions["default"].apply(lambda x: "Yes" if x==1 else "No")

        return dtf_predictions


    @staticmethod
    def write_excel(dtf_predictions):
        bytes_file = io.BytesIO()
        excel_writer = pd.ExcelWriter(bytes_file)
        dtf_predictions.to_excel(excel_writer, sheet_name='Sheet1', na_rep='', index=False)
        excel_writer.save()
        bytes_file.seek(0)
        return bytes_file
