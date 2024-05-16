import pickle
from sklearn.naive_bayes import GaussianNB, MultinomialNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC, LinearSVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import numpy as np
data_dict = pickle.load(open('./data.pickle', 'rb'))
data = np.asarray(data_dict['data'])
labels = np.asarray(data_dict['labels'])
x_train, x_test, y_train, y_test = train_test_split(data, labels, test_size=0.2, shuffle=True, stratify=labels)
# Train and Test with Naive Bayes algorithm
print('Naive Bayes Classifier')
nb = GaussianNB()
nb.fit(x_train, y_train)
y_predict = nb.predict(x_test)
score = accuracy_score(y_predict, y_test)
print('{}% of samples were classified correctly !'.format(score * 100))

# Train and Test with Multinomial Naive Bayes algorithm
print('Multinomial Naive Bayes Classifier')
mnb = MultinomialNB()
mnb.fit(x_train, y_train)
y_predict = mnb.predict(x_test)
score = accuracy_score(y_predict, y_test)
print('{}% of samples were classified correctly !'.format(score * 100))

# Train and Test with Random Forest algorithm
print('Random Forest Classifier')
rf = RandomForestClassifier()
rf.fit(x_train, y_train)
y_predict = rf.predict(x_test)
score = accuracy_score(y_predict, y_test)
print('{}% of samples were classified correctly !'.format(score * 100))

# Train and Test with SVM (Support Vector Machines) algorithm
print('SVM Classifier')
svm = SVC(kernel='linear', C=1, random_state=0)
svm.fit(x_train, y_train)
y_predict = svm.predict(x_test)
score = accuracy_score(y_predict, y_test)
print('{}% of samples were classified correctly !'.format(score * 100))

# Train and Test with Linear SVM (Support Vector Machines) algorithm
print('Linear SVM Classifier')
lsvm = LinearSVC(C=1, random_state=0)
lsvm.fit(x_train, y_train)
y_predict = lsvm.predict(x_test)
score = accuracy_score(y_predict, y_test)
print('{}% of samples were classified correctly !'.format(score * 100))

# Train and Test with KNN (K Nearest Neighbour) algorithm
print('KNN Classifier')
knn = KNeighborsClassifier(n_neighbors=5)
knn.fit(x_train, y_train)
y_predict = knn.predict(x_test)
score = accuracy_score(y_predict, y_test)
print('{}% of samples were classified correctly !'.format(score * 100))

# Train and Test with Logistic Regression algorithm
print('Logistic Regression Classifier')
lr = LogisticRegression(random_state=0)
lr.fit(x_train, y_train)
y_predict = lr.predict(x_test)
score = accuracy_score(y_predict, y_test)
print('{}% of samples were classified correctly !'.format(score * 100))

# Train and Test with Decision Tree algorithm
print('Decision Tree Classifier')
dt = DecisionTreeClassifier(random_state=0)
dt.fit(x_train, y_train)
y_predict = dt.predict(x_test)
score = accuracy_score(y_predict, y_test)
print('{}% of samples were classified correctly !'.format(score * 100))

# Save the best model to a pickle file



# Save the best model to a pickle file
# best_model = max([nb, rf, svm, knn], key=lambda x: accuracy_score(x.predict(x_test), y_test))
# f = open('model.p', 'wb')
# pickle.dump({'model': best_model}, f)
# f.close()
