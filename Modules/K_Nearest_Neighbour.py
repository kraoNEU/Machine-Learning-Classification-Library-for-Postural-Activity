from sklearn.preprocessing import StandardScaler
from sklearn import neighbors
from sklearn.metrics import f1_score, confusion_matrix, roc_auc_score
from sklearn import neighbors


class KNNClassifier:

    def train_model(self, x_train, x_test):
        scaler = StandardScaler()
        scaler.fit_transform(x_train)
        scaler.transform(x_test)

    def classifier(self, x_train, y_train, x_test, y_test):
        KNN_model = neighbors.KNeighborsClassifier(n_neighbors=30, n_jobs=-1)
        KNN_model.fit(x_train, y_train)
        y_test = y_test.squeeze()
        pred = KNN_model.predict(x_test)
        return pred

    def metrics(self, x_train, y_train, x_test):
        f1_list = []
        k_list = []
        for k in range(1, 10):
            clf = neighbors.KNeighborsClassifier(n_neighbors=k, n_jobs=-1)
            clf.fit(x_train, y_train)
            pred = clf.predict(x_test)
            f = f1_score(y_test, pred, average='macro')
            f1_list.append(f)
            k_list.append(k)