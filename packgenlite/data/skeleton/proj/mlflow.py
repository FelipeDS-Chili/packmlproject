import multiprocessing
import time
import warnings
import category_encoders as ce
import joblib
import mlflow
import pandas as pd
import os
from memoized_property import memoized_property
from mlflow.tracking import MlflowClient
from psutil import virtual_memory
from sklearn.compose import ColumnTransformer
from sklearn.svm import SVC
from sklearn.ensemble import GradientBoostingRegressor,RandomForestClassifier, RandomForestClassifier, VotingClassifier, AdaBoostClassifier , GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression, SGDClassifier, LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split, RandomizedSearchCV, GridSearchCV
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler, OrdinalEncoder, Binarizer, RobustScaler, MinMaxScaler
from termcolor import colored
from xgboost import XGBRegressor
from utils import compute_f1, simple_time_tracker, compute_precision, get_data_filled

from sklearn.impute import KNNImputer,SimpleImputer
from scipy.stats import uniform, randint
from xgboost import XGBClassifier
from imblearn.pipeline import make_pipeline
from imblearn.over_sampling import SMOTE, ADASYN
from imblearn.pipeline import Pipeline as Pipeline_imb
from sklearn.inspection import permutation_importance
from sklearn.naive_bayes import GaussianNB
from google.cloud import storage



MLFLOW_URI = "https://mlflow.lewagon.co/"

BUCKET_NAME = 'wagon-ml-felipeinostrozarios-21'

BUCKET_TRAIN_DATA_PATH = 'data/train_1k.csv'

MODEL_NAME = 'taxifare'

MODEL_VERSION = 'v1'

STORAGE_LOCATION = 'models/simpletaxifare/model_taxi.joblib'



class Trainer(object):
    ESTIMATOR = "LogisticRegression"
    EXPERIMENT_NAME = "Invesscience_batch_#463"
    IMPUTER = 'SimpleImputer'
    SCALER_AMOUNT = 'RobustScaler'
    SCALER_PROFESSIONALS = 'MinMaxScaler'
    SCALER_TIME = 'StandardScaler'
    SCALER_PARTICIPANTS = 'StandardScaler'

    def __init__(self, X, y, **kwargs):
        """
        FYI:
        __init__ is called every time you instatiate Trainer
        Consider kwargs as a dict containig all possible parameters given to your constructor
        Example:
            TT = Trainer(nrows=1000, estimator="Linear")
               ==> kwargs = {"nrows": 1000,
                            "estimator": "Linear"}
        :param X:
        :param y:
        :param kwargs:
        """

        self.pipeline = None
        self.kwargs = kwargs
        self.upload = self.kwargs.get("upload", True)
        self.local = kwargs.get("local", False)  # if True training is done locally
        self.smote = kwargs.get("smote", False)
        self.mlflow = kwargs.get("mlflow", False)
        self.tag = kwargs.get("tag_description", "nada")
        self.experiment_name = kwargs.get("experiment_name", self.EXPERIMENT_NAME)  # cf doc above
        self.model_params = None  # for
        self.grid_search_choice = kwargs.get("grid_search_choice", False)
        self.X_train = X
        self.y_train = y
        del X, y
        self.split = self.kwargs.get("split", True)  # cf doc above
        if self.split:
            self.X_train, self.X_val, self.y_train, self.y_val = train_test_split(self.X_train, self.y_train,
                                                                                  test_size=0.3)
        self.nrows = self.X_train.shape[0]  # nb of rows to train on
        self.log_kwargs_params()
        self.log_machine_specs()
        
        

    def get_estimator(self):
        estimator = self.kwargs.get("estimator", self.ESTIMATOR)
        
        if estimator == "LogisticRegression":
            model = LogisticRegression(class_weight= 'balanced')
        
        elif estimator == "SVC":
            model = SVC(class_weight='balanced')
        
        elif estimator == "KNeighborsClassifier":
            model = KNeighborsClassifier()
        
        elif estimator == "DecisionTree":
            model = DecisionTreeClassifier(class_weight ='balanced')


        elif estimator == "RandomForestClassifier":
            model = RandomForestClassifier()
            

        elif estimator == "xgboost":

            model = XGBClassifier()

        elif estimator == "GaussianNB":

            model = GaussianNB()

        elif estimator == "adaboost":
            model = AdaBoostClassifier()
            
        elif estimator =='SGDC':
            model = SGDClassifier(class_weight ='balanced')


        elif estimator =='voting':

            model_1 = SGDClassifier()
            model_2 = SGDClassifier()
            model_3 = SGDClassifier()
            model_4 = SGDClassifier()
            model_5 = SVC()
            model_6 = SVC()



            model = VotingClassifier(estimators=[('model1a', model_1),
                                                ('model2a', model_2),
                                                ('model4a', model_4),
                                                ('model5a', model_5),
                                                ('model6a', model_6)]
                                ,voting='soft')





        estimator_params = self.kwargs.get("estimator_params", {}) #Dictionary
        self.mlflow_log_param("estimator", estimator)
        model.set_params(**estimator_params)
        print(colored(model.__class__.__name__, "red"))
        return model

    def get_imputer(self):
        imputer = self.kwargs.get("imputer", self.IMPUTER)
        if imputer == "SimpleImputer":
            imputer_use = SimpleImputer()
        if imputer == "KNNImputer":
            imputer_use = KNNImputer()


        imputer_params = self.kwargs.get("imputer_params", {})
        self.mlflow_log_param("imputer", imputer)
        imputer_use.set_params(**imputer_params)
        print(colored(imputer_use.__class__.__name__, "blue"))

        return imputer_use



    def get_scaler_raised_amount(self):
        scaler_amount = self.kwargs.get("scaler_amount", self.SCALER_AMOUNT)
        if scaler_amount == "RobustScaler":
            scaler_use = RobustScaler()
        elif scaler_amount == "StandardScaler":
            scaler_use = StandardScaler()
        elif scaler_amount == 'MinMaxScaler':
            scaler_use = MinMaxScaler()

        scaler_amount_params = self.kwargs.get("scaler_amount_params", {})
        self.mlflow_log_param("scaler_amount", scaler_amount)
        scaler_use.set_params(**scaler_amount_params)
        print(colored(scaler_use.__class__.__name__, "blue"))

        return scaler_use


    def get_scaler_professionals(self):
        scaler_professionals = self.kwargs.get("scaler_professionals", self.SCALER_PROFESSIONALS)
        if scaler_professionals == "RobustScaler":
            scaler_use = RobustScaler()
        elif scaler_professionals == "StandardScaler":
            scaler_use = StandardScaler()
        elif scaler_professionals == 'MinMaxScaler':
            scaler_use = MinMaxScaler()

        scaler_professionals_params = self.kwargs.get("scaler_professionals_params", {})
        self.mlflow_log_param("scaler_professionals", scaler_professionals)
        scaler_use.set_params(**scaler_professionals_params)
        print(colored(scaler_use.__class__.__name__, "blue"))

        return scaler_use

    def get_scaler_time(self):
        scaler_time = self.kwargs.get("scaler_time", self.SCALER_TIME)
        if scaler_time == "RobustScaler":
            scaler_use = RobustScaler()
        elif scaler_time == "StandardScaler":
            scaler_use = StandardScaler()
        elif scaler_time == 'MinMaxScaler':
            scaler_use = MinMaxScaler()

        scaler_time_params = self.kwargs.get("scaler_time_params", {})
        self.mlflow_log_param("scaler_time", scaler_time)
        scaler_use.set_params(**scaler_time_params)
        print(colored(scaler_use.__class__.__name__, "blue"))

        return scaler_use


    def get_scaler_participant(self):
        scaler_participants = self.kwargs.get("scaler_participants", self.SCALER_PARTICIPANTS)
        if scaler_participants == "RobustScaler":
            scaler_use = RobustScaler()
        elif scaler_participants == "StandardScaler":
            scaler_use = StandardScaler()
        elif scaler_participants == 'MinMaxScaler':
            scaler_use = MinMaxScaler()

        scaler_participant_params = self.kwargs.get("scaler_participant_params", {})
        self.mlflow_log_param("scaler_participants", scaler_participants)
        scaler_use.set_params(**scaler_participant_params)
        print(colored(scaler_use.__class__.__name__, "blue"))

        return scaler_use


    def set_pipeline(self):

        categorical_features_1 = ['category_code', 'country_code','state_code', 'founded_at','timediff_founded_series_a', 'time_diff_series_a_now'] #first use imputer /after ohe
        categorical_features_2 = ['participants_a', 'raised_amount_usd_a', 'rounds_before_a', 'mean_comp_worked_before',  'founder_count', 'degree_count'] # impute first, after ordinals
        booleans_features = ['graduate', 'undergrad','professional', 'MBA_bool', 'cs_bool', 'phd_bool', 'top_20_bool', 'mean_comp_founded_before', 'female_ratio'] # ordinals/binaries


        #Defining imputers
        imputer = self.get_imputer()
        imputer_2 = SimpleImputer(strategy = 'most_frequent')

        #pipes for each feature

        pipe_1 = Pipeline([('imputer', imputer_2),
                            ('ohe', OneHotEncoder(handle_unknown='ignore'))])

        pipe_2 = Pipeline([('imputer_ord', imputer),
                                ('ord_encoder', OneHotEncoder(handle_unknown='ignore'))
                            ])

        pipe_bool =  Pipeline([('imputer_bool', imputer_2),
                                ('ord_encoder', OneHotEncoder(handle_unknown='ignore'))
                            ])
        #process

        feateng_blocks = [ ('cat_1', pipe_1, categorical_features_1),
                            ('cat_2',  pipe_2, categorical_features_2),
                            ('cat_bool', pipe_bool, booleans_features)]




        #Columntransformer keeping order
        preprocessor = ColumnTransformer(feateng_blocks, remainder= 'passthrough')

        #final_pipeline
        self.pipeline = Pipeline(steps = [('preprocessing', preprocessor),
                            ('model_use', self.get_estimator())] )


    

        if self.smote:

            smote =ADASYN(sampling_strategy = 'minority', n_neighbors= 20)
            self.pipeline =Pipeline_imb([
                ('prep',preprocessor),
                ('smote', smote),
                ('model_use', self.get_estimator())])


        # Random search
        if self.grid_search_choice:
            grid_search = RandomizedSearchCV(
                self.pipeline,
                param_distributions ={

                "model_use__learning_rate": uniform(0,1),
               "model_use__gamma" : uniform(0,2),
               "model_use__max_depth": randint(1,15),
               "model_use__colsample_bytree": randint(0.1,1),
               "model_use__subsample": [0.2, 0.4, 0.5],
               "model_use__reg_alpha": uniform(0,1),
               "model_use__reg_lambda": uniform(1,10),
               "model_use__min_child_weight": randint(1,10),
               "model_use__n_estimators": randint(1000,3000)

                    },  #param depending of the model to use
                cv=35,
                scoring='f1',
                n_iter = 10,
                n_jobs = -1 )


            grid_search.fit(self.X_train, self.y_train)

            self.pipeline = grid_search.best_estimator_
            self.grid_params = grid_search.get_params

            self.set_tag('model_used', self.pipeline)



    @simple_time_tracker
    def train(self):
        tic = time.time()
        self.set_pipeline()
        self.pipeline.fit(self.X_train, self.y_train)
        # mlflow logs
        self.mlflow_log_metric("train_time", int(time.time() - tic))
        self.set_tag('tag_instance', self.tag)

    def evaluate(self):
        f1_train = self.compute_f1(self.X_train, self.y_train)
        precision_train = self.compute_precision(self.X_train, self.y_train)

        self.mlflow_log_metric("f1score_train", f1_train)
        self.mlflow_log_metric("precision_train", precision_train)

        if self.split:
            f1_val = self.compute_f1(self.X_val, self.y_val, show=True)
            precision_val = self.compute_precision(self.X_val, self.y_val, show=True)
            self.mlflow_log_metric("f1score_val", f1_val)
            self.mlflow_log_metric("precision_val", precision_val)
            print(colored("f1 train: {} || f1 val: {}".format(f1_train, f1_val), "yellow"))
            print(colored("precision train: {} || precision val: {}".format(precision_train, precision_val), "yellow"))
        else:
            print(colored("f1 train: {}".format(f1_train), "blue"))
            print(colored("precision train: {}".format(precision_train), "blue"))


    def compute_f1(self, X_test, y_test, show=False):
        if self.pipeline is None:
            raise ("Cannot evaluate an empty pipeline")
        y_pred = self.pipeline.predict(X_test)
        if show:
            res = pd.DataFrame(y_test)
            res["pred"] = y_pred
            print(colored(res.sample(self.y_val.shape[0]), "blue")) #Aumentar tamaño de muestra de validacion
        f1 = compute_f1(y_pred, y_test)
        return round(f1, 3)


    def compute_precision(self, X_test, y_test, show=False):
        if self.pipeline is None:
            raise ("Cannot evaluate an empty pipeline")
        y_pred = self.pipeline.predict(X_test)
        if show:
            res = pd.DataFrame(y_test)
            res["pred"] = y_pred
            print(colored(res.sample(self.y_val.shape[0]), "blue")) #Aumentar tamaño de muestra de validacion
        precision = compute_precision(y_pred, y_test)
        return round(precision, 3)



    def save_model(self, upload=True, auto_remove=True):
        """Save the model into a .joblib and upload it on Google Storage /models folder
        HINTS : use sklearn.joblib (or jbolib) libraries and google-cloud-storage"""
        
        joblib.dump(self.pipeline, 'model_taxi.joblib')
        print(colored("model_taxi.joblib saved locally", "green"))
        if self.upload:
            self.upload_model_to_gcp()
            print(f"uploaded model_taxi.joblib to gcp cloud storage under \n => {STORAGE_LOCATION}")

    #  optimization of integers and floats (downcast)
    def df_optimized(self, df, verbose=True, **kwargs):
        """
        Reduces size of dataframe by downcasting numeircal columns
        :param df: input dataframe
        :param verbose: print size reduction if set to True
        :param kwargs:
        :return: df optimized
        """
        in_size = df.memory_usage(index=True).sum()
        # Optimized size here
        for type in ["float", "integer"]:
            l_cols = list(df.select_dtypes(include=type))
            for col in l_cols:
                df[col] = pd.to_numeric(df[col], downcast=type)
                if type == "float":
                    df[col] = pd.to_numeric(df[col], downcast="integer")
        out_size = df.memory_usage(index=True).sum()
        ratio = (1 - round(out_size / in_size, 2)) * 100
        GB = out_size / 1000000000
        if verbose:
            print("optimized size by {} % | {} GB".format(ratio, GB))
        return df

    # method to upload the model to gcp
    def upload_model_to_gcp(self):

        client = storage.Client()

        bucket = client.bucket(BUCKET_NAME)

        blob = bucket.blob(STORAGE_LOCATION)

        blob.upload_from_filename('model.joblib')


    ### MLFlow methods
    @memoized_property
    def mlflow_client(self):
        mlflow.set_tracking_uri(MLFLOW_URI)
        return MlflowClient()

    @memoized_property
    def mlflow_experiment_id(self):
        try:
            return self.mlflow_client.create_experiment(self.experiment_name)
        except BaseException:
            return self.mlflow_client.get_experiment_by_name(self.experiment_name).experiment_id

    @memoized_property
    def mlflow_run(self):
        return self.mlflow_client.create_run(self.mlflow_experiment_id)

    def mlflow_log_param(self, key, value):
        if self.mlflow:
            self.mlflow_client.log_param(self.mlflow_run.info.run_id, key, value)

    def mlflow_log_metric(self, key, value):
        if self.mlflow:
            self.mlflow_client.log_metric(self.mlflow_run.info.run_id, key, value)

    def log_estimator_params(self):
        reg = self.get_estimator()
        self.mlflow_log_param('estimator_name', reg.__class__.__name__)
        params = reg.get_params()
        for k, v in params.items():
            self.mlflow_log_param(k, v)

    def set_tag(self, key, value):
        if self.mlflow:
            self.mlflow_client.set_tag(self.mlflow_run.info.run_id, key, value)

    def log_kwargs_params(self):
        if self.mlflow:
            for k, v in self.kwargs.items():
                self.mlflow_log_param(k, v)

    def log_machine_specs(self):
        cpus = multiprocessing.cpu_count()
        mem = virtual_memory()
        ram = int(mem.total / 1000000000)
        self.mlflow_log_param("ram", ram)
        self.mlflow_log_param("cpus", cpus)


 ################------------ RUN THE CODE ------#####################################################################


if __name__ == "__main__":
    warnings.simplefilter(action='ignore', category=FutureWarning)

    # Get and clean data
    experiment = "Invesscience_batch_#463"

    #Change the reference HERE !!!

    df = get_data()
    y_train = df["target"]
    X_train = df.drop(columns =['target']) #Change when we have categorical var




    for i in range(1):
        for estimator_iter in ['voting'
                              #  'SGDC'
                               #'xgboost',
                                #'GradientBoostingClassifier',
                                #'LogisticRegression'
                                #'SVC',
                                 #'adaboost',
                                 #'DecisionTree'
                                # 'RandomForestClassifier'
                                 ]:

            params = dict(tag_description=f'[MODEL FINAL]{estimator_iter}][{year}][{reference}]' ,estimator = estimator_iter,
                estimator_params ={ 'weights' :[6, 2, 5, 3, 4]},
                local=False, split=True,  mlflow = True, experiment_name=experiment,
                imputer= 'SimpleImputer', imputer_params = {'strategy': 'most_frequent'},
                grid_search_choice= False, smote=True, upload=True) #agregar




            print("############   Loading Data   ############")


            #del df
            print("shape: {}".format(X_train.shape))
            print("size: {} Mb".format(X_train.memory_usage().sum() / 1e6))
            # Train and save model, locally and
            t = Trainer(X=X_train, y=y_train, **params)
            #del X_train, y_train


            print(colored("############  Training model   ############", "red"))
            t.train()
            print(colored("############  Evaluating model ############", "blue"))
            t.evaluate()
            print(colored("############   Saving model    ############", "green"))
            t.save_model()


 ################------------Params founded ------#####################################################################
