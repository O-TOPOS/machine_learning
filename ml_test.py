import numpy as np
from sklearn import tree, label
import psycopg2 as ps

class Database:

    def __init__(self):

        self.connection = None
        self.cursor = None

    def connect(self):
        try:
            self.connection = ps.connect(
                database='postgres',
                user='postgres',
                host='localhost',
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


def get_features_and_labels(table):
    features = []
    labels = []
    for record in get_table_data("%s" % table ):
        features.append(record[3:])
        labels.append(record[2])
    return features, labels


features, labels = get_features_and_labels('ml_dataset')

le = LabelEncoder()


# Initialise classifier
clf = tree.DecisionTreeClassifier()
# Train classifier with features and their corresponding labels
clf = clf.fit(features, labels)