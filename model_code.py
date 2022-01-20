import pandas as pd
import numpy as np
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error
from sklearn.linear_model import LinearRegression
import xgboost as xgb
from keras.models import Sequential
from keras.layers import LSTM
from keras.layers import Dense
from keras.layers import RepeatVector
from keras.layers import TimeDistributed
from keras.layers import Masking
from sklearn.preprocessing import PolynomialFeatures

# This is the value I replaced categorical variables with
MISSING = "MISSING"

# This function gets the columns to use for the model
def get_columns_to_keep(training_data):
    # Here I define which columns to keep
    cols_to_keep = []
    for col in training_data.columns:
        # After investigation, I realized the model does best when only using continuous variables
        if np.issubdtype(training_data[col].dtype, np.number):

            correlation = training_data['job_performance'].corr(training_data[col].fillna(training_data[col].mean()))

            # At one point I tried only keeping variables with a certain correlation, but again the model did better
            # with all variables.

            # print(col + " has correlation of {} with job_performance".format(correlation))
            cols_to_keep.append(col)
            # if correlation >= 0.01:
            #     cols_to_keep.append(col)


    return cols_to_keep

# This function fills in the NA values
def process_data(data, columns):
    # Fill in NA
    for col in columns:
        # If the column is all null I filled with 0's
        if sum(data[col].isnull()) == len(data[col]):
            data[col] = 0
        elif np.issubdtype(data[col].dtype, np.number):
            # I chose to fill NA values with the mean
            data[col] = data[col].fillna(data[col].mean())

    data = data[columns]

    # I tried using dummy variables on the categorical variables but the model did not perform as well
    #data = pd.get_dummies(data, columns=categorical_columns, prefix=categorical_columns)

    return data

# I created this function when I was going to create features, but the model did well without it so it's empty
def create_features(data):
    return None

# This function splits the data into x and y data
def split_data(data):
    x_data = data.drop('job_performance', axis=1)
    y_data = data['job_performance']

    return (x_data, y_data)

# This is the function to actually train the model
def train_model(x_train, y_train):
    # Here I tried a linear regression model
    # model = LinearRegression().fit(x_train, y_train)

    # Here I do polynomial expansion on the featureset
    poly = PolynomialFeatures(2)
    x_train = poly.fit_transform(x_train.to_numpy())

    # Here I prepare the data and train an XGBoost model
    dtrain = xgb.DMatrix(x_train, label=y_train)
    param = {'max_depth': 20, 'eta': 0.4, 'objective': 'reg:squarederror', 'booster':'gbtree'}
    param['eval_metric'] = 'rmse'
    param['nthread'] = 4

    model = xgb.train(param, dtrain)


    # Here was a basic Deep Neural Network I tried

    # model = Sequential()
    # model.add(Dense(30, input_dim=len(x_train.columns), activation='relu'))
    # model.add(Dense(15, activation='sigmoid'))
    # model.add(Dense(1, activation='sigmoid'))
    #
    # model.compile(loss='mean_squared_error', optimizer='sgd')
    #
    # model.fit(x_train, y_train, epochs=50)

    return model

# This function predicts the y values by doing polynomial expansion + prepping for XGBoost
def predict_values(model, x_test):
    poly = PolynomialFeatures(2)
    x_test = poly.fit_transform(x_test)
    x_test = xgb.DMatrix(x_test)

    return model.predict(x_test)

# This function does 10-fold cross validation on the data
def evaluate_model(x_train, y_train):
    cv = KFold(n_splits=10, random_state=42, shuffle=False)
    all_mean_squared_errors = []
    fold = 1
    for train_index, test_index in cv.split(x_train):
        print("Train Index: ", train_index, "\n")
        print("Test Index: ", test_index)

        x_fold_train, x_fold_test, y_fold_train, y_fold_test = x_train.iloc[train_index], x_train.iloc[test_index],\
                                                               y_train.iloc[train_index], y_train.iloc[test_index]


        fold_model = train_model(x_fold_train, y_fold_train)
        fold_predictions = predict_values(fold_model, x_fold_test)
        mse = mean_squared_error(y_fold_test, fold_predictions)
        print("MSE on Fold {} is {}".format(fold, mse))
        all_mean_squared_errors.append(mse)
        fold += 1

    return np.mean(all_mean_squared_errors)



if __name__ == "__main__":
    # Read in the data
    training_data = pd.read_csv("trainingset.csv")
    test_data = pd.read_csv("testset.csv")

    # Keep the columns of interest
    cols_to_keep = get_columns_to_keep(training_data)
    test_cols_to_keep = cols_to_keep.copy()
    test_cols_to_keep.remove('job_performance')

    # Process the test and training data
    training_data = process_data(training_data, cols_to_keep)
    test_data = process_data(test_data, test_cols_to_keep)

    # Split up the training data into x and y
    x_train, y_train = split_data(training_data)

    # Evaluate the model using 10-fold CV
    mean_squared_error = evaluate_model(x_train=x_train, y_train=y_train)

    # Print the MSE of the 10-fold CV which is: 36315.59
    print(mean_squared_error)

    # Create the full model using all of the training data
    full_model = train_model(x_train, y_train)

    # Make predictions using the full model
    predictions = predict_values(full_model, test_data)

    # Read the test data as is and simply fill in the prediction column and then save the file as my submission file
    results = pd.read_csv("testset.csv")
    results['job_performance'] = predictions
    results.to_csv("testset.csv")