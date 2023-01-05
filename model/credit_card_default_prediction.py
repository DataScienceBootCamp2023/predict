import pandas as pd
import io
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.pipeline import Pipeline

class CreditCardDefaultPrediction():
    def __init__(self, dtf_input):
      self.dtf_input = dtf_input

    @staticmethod
    def predict(self, model_name):
    
            # Load the training dataset from the CSV file
        self.df = pd.read_csv('UCI_Credit_Card.csv')

        # Renaming the target column for better reference. 
        self.df.rename(columns = {"ID" : "id","LIMIT_BAL" : "limit_bal", "SEX" : "sex", "EDUCATION" : "education", "MARRIAGE" : "marriage","AGE":"age",
                             "PAY_0":"pay_1", "PAY_2":"pay_2", "PAY_3":"pay_3", "PAY_4":"pay_4","PAY_5":"pay_5", "PAY_6":"pay_6","BILL_AMT1":"bill_amt_1", 
                             "BILL_AMT2":"bill_amt_2","BILL_AMT3":"bill_amt_3", "BILL_AMT4":"bill_amt_4", "BILL_AMT5":"bill_amt_5",
                             "BILL_AMT6":"bill_amt_6", "PAY_AMT1":"pay_amt_1", "PAY_AMT2":"pay_amt_2", "PAY_AMT3":"pay_amt_3", "PAY_AMT4":"pay_amt_4", 
                             "PAY_AMT5":"pay_amt_5", "PAY_AMT6":"pay_amt_6", "default.payment.next.month":"default"
                             }, inplace = True)

        # Split the data into features and target
        self.X = self.df.drop(["default", "id"], axis=1)
        self.y = self.df["default"]

        # Split the data into train and test sets
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        # Initialize the model with the training data
       # model = CreditCardDefaultPrediction(X_train, y_train)
       # Define the feature engineering pipeline with Feature transformation(Scaling), Feature construction, Feature extraction(PCA)
        feature_engineering_pipeline = Pipeline([
            ('scaler', StandardScaler()),
            ('constructor', FeatureConstructor()),
            ('pca', PCA()),
            ('model', model)
        ])
        # Transform the features using the pipeline
        X_train_transformed = feature_engineering_pipeline.fit_transform(self.X_train)
        X_test_transformed = feature_engineering_pipeline.transform(self.X_test)

        # Fit the model on the transformed features
        model.fit(X_train_transformed, self.y_train) 
            # Make predictions on the input data
        predictions = model.predict(self.dtf_input.iloc[:, 1:-1])
        # Create a dataframe with the predictions and the original data
        dtf_predictions = pd.concat([self.dtf_input.iloc[:, 0], pd.DataFrame(predictions, columns=["default"])], axis=1)
        dtf_predictions["default"] = dtf_predictions["default"].apply(lambda x: "Yes" if x==1 else "No"

        return dtf_predictions

    @staticmethod
    def write_excel(dtf_predictions):
        bytes_file = io.BytesIO()
        excel_writer = pd.ExcelWriter(bytes_file)
        dtf_predictions.to_excel(excel_writer, sheet_name='Sheet1', na_rep='', index=False)
        excel_writer.save()
        bytes_file.seek(0)
        return bytes_file


