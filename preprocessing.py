import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE

def preprocess_data(filename, datasetid):
    data = pd.read_csv(filename)
    # print(data.head(10))

    data = data.fillna(value=False)
    # print(data.head(10))
    # print(data.tail(10))
    # data = data.dropna()
    X = pd.DataFrame(data.drop(['defects'], axis=1))
    y = pd.DataFrame(data['defects'])
    y *= 1
    
    sm = SMOTE(random_state=1234, k_neighbors=5) #for oversampling minority data
    X, y = sm.fit_resample(X, y)
    X = pd.DataFrame(X, columns=X.columns)
    y = pd.DataFrame(y, columns=y.columns)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=1234)
    X_train, X_validation, y_train, y_validation = train_test_split(X_train, y_train, test_size=0.20, random_state=1234)

    total_data = X_train.copy()
    total_data['defects'] = y_train

    corr = total_data.corr()
    sns.set(font_scale=0.5)
    plt.margins()
    sns.heatmap(corr, xticklabels=1, yticklabels=1)
    plt.savefig('./fig/correlation' + str(datasetid) + '.jpg')
    # plt.show()

    return X, y, X_train, X_test, X_validation, y_train, y_test, y_validation

preprocess_data('./Data/pc1.csv', 10)