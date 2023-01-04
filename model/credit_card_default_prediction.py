import pandas as pd
#import numpy as np
#from sklearn import feature_extraction, metrics
import io
#from tqdm import tqdm
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
#from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.pipeline import Pipeline
from sklearn.base import BaseEstimator, TransformerMixin

class CreditCardDefaultPrediction():
    def __init__(self, dtf_input):
      self.dtf_input = dtf_input
      
    def __init__(self):
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
        self.X = self.df.drop(["default","id"], axis=1)
        #Get Target data 
        self.y = self.df[""default""]
         # separate dataset into train and test
        self.X_train, self.X_test, self.y_train, self.y_test = self.train_test_split(self.X,self.y,test_size=0.2,random_state=42)
       # Transform the numerical columns as those are not having Gaussian Distribution.
        transformer = ColumnTransformer([("Transformed", PowerTransformer(), num_columns)],remainder="passthrough" )
        transformer.fit(self.X_train)

        columns = num_columns+cat_columns
        self.X_train = self.pd.DataFrame(transformer.transform(self.X_train), columns = columns)
        self.X_test = self.pd.DataFrame(transformer.transform(self.X_test), columns = columns) 
        
    class FeatureConstructor(BaseEstimator, TransformerMixin):
            def __init__(self):
                pass

            def fit(self, self.X, self.y=None):
                return self

            def transform(self, self.X, self.y=None):
                # Construct your new features here and return the modified DataFrame
                return self.X

    def apply_feature_engineering(self.X_train, self.y_train, self.X_test, self.y_test, model):
            # Define the feature engineering pipeline with Feature transformation(Scaling), Feature construction, Feature extraction(PCA)
            feature_engineering_pipeline = Pipeline([
                ('scaler', StandardScaler()),
                ('constructor', FeatureConstructor()),
                ('pca', PCA())
            ])
            
            # Transform the features using the pipeline
            self.X_train_transformed = feature_engineering_pipeline.fit_transform(self.X_train)
            self.X_test_transformed = feature_engineering_pipeline.transform(self.X_test)
            
            # Fit the model on the transformed features
            self.model.fit(self.X_train_transformed, self.y_train)
            
            # Make predictions on the transformed test set
            self.y_pred = self.model.predict(self.X_test_transformed)
            # Define a list of models that you want to apply feature engineering to
            model = [
                #DecisionTreeClassifier(),
                LogisticRegression()
               # KNeighborsClassifier(),
               #GradientBoostingClassifier(),
               #AdaBoostClassifier(),
               #RandomForestClassifier(),
             
                #SVC()
            ]


     

    @staticmethod
    def predict(self, top=1):
        ## process data
        lst_input = list(set( self.dtf_input.iloc[:,1:-1].tolist() ))
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
        self.X = self.df.drop(["default","id"], axis=1)
        #Get Target data 
        self.y = self.df[""default""]
         # separate dataset into train and test
        self.X_train, self.X_test, self.y_train, self.y_test = self.train_test_split(self.X,self.y,test_size=0.2,random_state=42)
       # Transform the numerical columns as those are not having Gaussian Distribution.
        transformer = ColumnTransformer([("Transformed", PowerTransformer(), num_columns)],remainder="passthrough" )
        transformer.fit(self.X_train)

        columns = num_columns+cat_columns
        self.X_train = self.pd.DataFrame(transformer.transform(self.X_train), columns = columns)
        self.X_test = self.pd.DataFrame(transformer.transform(self.X_test), columns = columns) 
        
            class FeatureConstructor(BaseEstimator, TransformerMixin):
                    def __init__(self):
                        pass

                    def fit(self, self.X, self.y=None):
                        return self

                    def transform(self, self.X, self.y=None):
                        # Construct your new features here and return the modified DataFrame
                        return self.X

            def apply_feature_engineering(self.X_train, self.y_train, self.X_test, self.y_test, model):
                    # Define the feature engineering pipeline with Feature transformation(Scaling), Feature construction, Feature extraction(PCA)
                    feature_engineering_pipeline = Pipeline([
                        ('scaler', StandardScaler()),
                        ('constructor', FeatureConstructor()),
                        ('pca', PCA())
                    ])
                    
                    # Transform the features using the pipeline
                    self.X_train_transformed = feature_engineering_pipeline.fit_transform(self.X_train)
                    self.X_test_transformed = feature_engineering_pipeline.transform(self.X_test)
                    
                    # Fit the model on the transformed features
                    self.tr_model=self.model.fit(self.X_train_transformed, self.y_train)
                    
                    # Make predictions on the transformed test set
                    self.y_pred = self.tr_model.predict(self.X_test_transformed)
                    # Define a list of models that you want to apply feature engineering to
                    model = [
                        #DecisionTreeClassifier(),
                        LogisticRegression()
                       # KNeighborsClassifier(),
                       #GradientBoostingClassifier(),
                       #AdaBoostClassifier(),
                       #RandomForestClassifier(),
                     
                        #SVC()
                    ]
              
        
        
        ## fit model
        #model = LogisticRegression()
        #model.fit(dtf_input.iloc[:, :-1], dtf_input.iloc[:, -1])

        ## make predictions
       # predictions = self.model.predict(self.X_test_transformed)
        predictions = self.tr_model.predict(dtf_input.iloc[:, 1:-1])
        dtf_predictions = pd.DataFrame(predictions, columns=["default"])
        dtf_predictions["default"] = dtf_predictions["default"].apply(lambda x: "Yes" if x==1 else "No")

        return dtf_predictions


    @staticmethod
    def write_excel(dtf_predictions):
        bytes_file = io.BytesIO()
        excel_writer = pd.ExcelWriter(bytes_file)
        dtf_predictions.to_excel(excel_writer, sheet_name='Sheet2', na_rep='', index=False)
        excel_writer.save()
        bytes_file.seek(0)
        return bytes_file
