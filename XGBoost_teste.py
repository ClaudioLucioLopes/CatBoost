import pandas
import numpy as np
import catboost as cb

# read in the train and test data from csv files
colnames = ['age' , 'wc' , 'fnlwgt' , 'ed' , 'ednum' , 'ms' , 'occ' , 'rel' , 'race' , 'sex' , 'cgain' , 'closs' ,
            'hpw' , 'nc' , 'label']
train_set = pandas.read_csv ("C:\\Users\\claud\\Downloads\\adult.data.csv" , header=None , names=colnames ,
                             na_values='?')
test_set = pandas.read_csv ("C:\\Users\\claud\\Downloads\\adult.test.csv" , header=None , names=colnames ,
                            na_values='?' , skiprows=[0])

# convert categorical columns to integers
category_cols = ['wc' , 'ed' , 'ms' , 'occ' , 'rel' , 'race' , 'sex' , 'nc' , 'label']
for header in category_cols:
    train_set[header] = train_set[header].astype ('category').cat.codes
    test_set[header] = test_set[header].astype ('category').cat.codes

# split labels out of data sets
train_label = train_set['label']
train_set = train_set.drop ('label' , axis=1)  # remove labels
test_label = test_set['label']
test_set = test_set.drop ('label' , axis=1)  # remove labels

# train default classifier
clf = cb.CatBoostClassifier ()
cat_dims = [train_set.columns.get_loc (i) for i in category_cols[:-1]]
clf.fit (train_set , np.ravel (train_label) , cat_features=cat_dims)
res = clf.predict (test_set)
print ('error:' , 1 - np.mean (res == np.ravel (test_label)))

from pandas_ml import ConfusionMatrix

test_set['label']

confusion_matrix = ConfusionMatrix (np.ravel (test_label) , res)
print ("Confusion matrix:\n%s" % confusion_matrix)

confusion_matrix.print_stats()

import matplotlib.pyplot as plt

confusion_matrix.plot ()
plt.show ()
