from keras.models import Sequential
from keras.layers import Dense, Dropout, Conv2D, Flatten, MaxPool2D
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder

# convolutional neural network classifier
def CNN(X, X_train, X_validation, y_train, y_validation):
    X_train_matrix = X_train.values
    X_validation_matrix = X_validation.values
    y_train_matrix = y_train.values
    y_validation_matrix = y_validation.values

    img_rows = 1
    img_cols = len(X.columns)

    X_train_f = X_train_matrix.reshape(X_train_matrix.shape[0], img_rows, img_cols, 1)
    X_validation_f = X_validation_matrix.reshape(X_validation_matrix.shape[0], img_rows, img_cols, 1)

    input_shape = (img_rows, img_cols, 1)


    model = Sequential()
    model.add(Conv2D(64, kernel_size=1, activation='relu', input_shape=input_shape))
    model.add(Conv2D(32, kernel_size=1, activation='relu'))
    model.add(Conv2D(16, kernel_size=1, activation='relu'))
    model.add(Conv2D(16, kernel_size=1, activation='relu'))
    model.add(Flatten())
    model.add(Dense(8, activation='relu'))
    model.add(Dense(1, activation='sigmoid'))

    model.compile(loss='binary_crossentropy',
                    optimizer='adam',
                    metrics=['accuracy'])

    model.fit(X_train_f, y_train_matrix, epochs=100, batch_size=35, validation_data=(X_validation_f, y_validation_matrix), verbose=0)
    return model

# random forest classifer
def random_forest(X_train, y_train):
    clf = RandomForestClassifier(n_estimators=100, max_depth=6, random_state=1234)
    clf.fit(X_train, y_train)
    return clf

# support vector machine classifier
def SVM(X_train, y_train):
    scaler = MinMaxScaler(feature_range=(-1, 1)).fit(X_train)
    X_train = scaler.transform(X_train)
    clf = SVC(kernel='rbf', degree=3, gamma='auto')
    clf.fit(X_train, y_train)
    return clf