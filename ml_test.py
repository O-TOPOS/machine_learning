import numpy as np
from sklearn import tree
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import pydotplus
from IPython.display import Image
import pydot
import psycopg2 as ps

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

def get_table_data(table):
    db = Database()
    db.connect()
    sql = "SELECT * FROM %s;" % table
    db.cursor.execute(sql)
    return db.cursor.fetchall()

def get_table_header(table):
    db = Database()
    db.connect()
    db.cursor.execute("Select * FROM %s" % table)
    return [desc[0] for desc in db.cursor.description]

def get_distinct_values(table, column):
    db = Database()
    db.connect()
    db.cursor.execute("SELECT DISTINCT %s FROM %s" % (column, table))
    values = []
    data = db.cursor.fetchall()
    for v in data:
        values.append(v[0])
    return values

def get_encoder(data, idx):
    feature_set = set() # Set of unique feature values
    for record in data:
        value = record[idx]
        if value != None:
            feature_set.add(record[idx])
        else:
            feature_set.add('') # Convert None to empty string
    # Encode values
    encoder = LabelEncoder()
    encoder.fit(list(feature_set))

    return encoder

def get_features_and_labels(table, id_col_idx, class_col_idx):

    # Data containing both features and labels
    data = get_table_data("%s" % table)

    ## Generate encoders for string features
    encoders = {}
    for i, v in enumerate(data[0]):
        # if i != id_col_idx:
        #     x_name = get_table_header()

        if type(data[0][i]) == str and i != id_col_idx: # select string columns and ignore the id column
            encoders[i] = get_encoder(data, i)

    # Fetch and transform features and labels
    xs = []  # list of features
    ys = []  # list of labels

    for record in data:
        if any(value is None for value in record):
            pass # discard any record with None numerical values
        else:

            x = [] # features for current record
            y = None # label of current record

            for i, v in enumerate(record):

                # Transforming the feature or label if it is a string
                if i in encoders.keys():
                    coded_value = encoders[i].transform([v,])
                    if i != class_col_idx and i != id_col_idx:
                        x.append(coded_value)
                    elif i == class_col_idx:
                        y = coded_value

                # Otherwise just add the numerical value
                else:
                    if i != class_col_idx and i != id_col_idx:
                        x.append(v)
                    elif i == class_col_idx:
                        y = v

            # Add features and label to their lists
            xs.append(x)
            ys.append(y)

    return xs, ys

def main():
    # Get features and labels
    xs, ys = get_features_and_labels('test_data.ml_features', 0, 1)
    x_names = get_table_header('test_data.ml_features')

    y_names = get_distinct_values('test_data.ml_features', 'classification')
    print(y_names)
    # Split data in a into training dataset and a test dataset
    x_train, x_test, y_train, y_test = train_test_split(xs, ys, test_size=0.3)

    # Initialise classifier
    clf = tree.DecisionTreeClassifier()

    # Train classifier with features and their corresponding labels
    clf = clf.fit(x_train, y_train)

    # Predict the label of the test dataset
    prediction = clf.predict(x_test)

    # Compare the accuracy of the prediction with the actual labels
    print(accuracy_score(y_test, prediction))

    # Visualising tree
    tree.export_graphviz(
        clf,
        out_file='tree_viz', feature_names=x_names, class_names=y_names

    )
    # graph = pydotplus.graph_from_dot_data(dot_data)
    # graph.write_pdf("iris.pdf")

main()
