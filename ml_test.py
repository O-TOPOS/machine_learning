import numpy as np
from sklearn import tree #label
import math
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

# le = LabelEncoder()

from sklearn.model_selection import train_test_split

accuracies = []
for i in range(10):
    f_train, f_test, l_train, l_test = train_test_split(features, labels, test_size=0.3)

# Initialise classifier
clf = tree.DecisionTreeClassifier()
# Train classifier with features and their corresponding labels
clf = clf.fit(f_train, l_train)

predictions = clf.predict(f_test, l_test)

from sklearn.metrics import accuracy_score

print(accuracy_score(l_test, predictions))

accuracies.append(accuracy_score(l_test, predictions))

import numpy

score = numpy.mean(accuracies)

