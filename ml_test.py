import numpy as np
from sklearn import tree
from sklearn import cluster
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, LabelBinarizer, MinMaxScaler, StandardScaler, Normalizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import pydotplus
from IPython.display import Image
import pydot
import psycopg2 as ps
import math
import pandas

# Variables
table_name = 'test_data.ml_features_jbg'
cat_columns = ['zonal_use', 'fp_02mtol', 'steepness', 'adj_class']
num_columns = ['density', 'fparea', 'gutterheight', 'roofspan', 'roofcount']
target_column = 'edb_class'


class Database:

    def __init__(self):

        self.connection = None
        self.cursor = None

    def connect(self):
        try:
            self.connection = ps.connect(
                database='edb_dev',
                user='postgres',
                host='amspcgqxh5y1',
                port=5432,
                password='admin')

            self.cursor = self.connection.cursor()

        except Exception as e:
            print('Cannot connect to the server!')

    def close(self):
        try:
            self.connection.close()
        except Exception as e:
            print('Cannot close the connection to the server!')


class Table:

    def __init__(self, name):
        self.db = Database()
        self.name = name

    def get_table_data(self):
        self.db.connect()
        sql = "SELECT * FROM %s;" % self.name
        self.db.cursor.execute(sql)
        return self.db.cursor.fetchall()

    def get_table_header(self):
        self.db.connect()
        self.db.cursor.execute("Select * FROM %s" % self.name)
        return [desc[0] for desc in self.db.cursor.description]

    def get_distinct_values(self, column):
        self.db.connect()
        self.db.cursor.execute("SELECT DISTINCT %s FROM %s" % (column, self.name))
        values = []
        data = self.db.cursor.fetchall()
        for v in data:
            values.append(v[0])
        return values


def get_dataframe(table_name):
    table = Table(table_name)
    data = table.get_table_data()
    columns = table.get_table_header()
    return pandas.DataFrame(data=data, columns=columns)

def get_features(dataframe, num_columns, cat_columns):
    df_num = dataframe[num_columns]
    df_num = StandardScaler().fit_transform(df_num) # Scale numerical data
    df_num = pandas.DataFrame(data=df_num, columns=num_columns) # Translating array back into a dataframe
    df_cat = dataframe[cat_columns]
    df_cat = pandas.get_dummies(data=df_cat) # Transform categorical data into dummy data
    df_all = pandas.concat([df_num,df_cat], axis=1, join_axes=[df_num.index]) # returned joined dataframe
    return df_all

def get_labels(dataframe, target_column):
    df_target = dataframe[target_column]
    return LabelBinarizer().fit_transform(df_target)

def predict():
    # Get the data
    df = get_dataframe(table_name)
    x = get_features(df, num_columns, cat_columns)
    y = get_labels(df, target_column)

    # Split data in a into training dataset and a test dataset
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.5)

    # Initialise classifier
    clf = tree.DecisionTreeClassifier()

    # Train classifier with features and their corresponding labels
    clf = clf.fit(x_train, y_train)

    # Predict the label of the test dataset
    prediction = clf.predict(x_test)

    # Compare the accuracy of the prediction with the actual labels
    return accuracy_score(y_test, prediction)

def main(run_number):
    predictions = []
    for i in range(run_number):
        predictions.append(predict())

    print(sum(predictions)/len(predictions))


main(10)
