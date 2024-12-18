import pickle
import numpy as np
import os
import matplotlib.pyplot as plt
from sklearn import svm
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB
import util

class HandGestureClassifier:
    def __init__(self, dataset_path):
        with open(dataset_path, 'rb') as f:
            dataset = pickle.load(f)
        self.features = dataset['features']
        self.labels = dataset['labels']
        self.classifiers = {
            'SVM': svm.LinearSVC(random_state=util.random_seed),
            'KNN': KNeighborsClassifier(n_neighbors=7),
            'Decision Tree': DecisionTreeClassifier(),
            'Random Forest': RandomForestClassifier(),
            'Naive Bayes': GaussianNB()
        }

    def evaluate_classifiers(self, test_size=0.2, random_state=42):
        X_train, X_test, y_train, y_test = train_test_split(self.features, self.labels, test_size=test_size, random_state=random_state)
        results = {}
        for name, clf in self.classifiers.items():
            clf.fit(X_train, y_train)
            y_pred = clf.predict(X_test)
            results[name] = {
                'accuracy': accuracy_score(y_test, y_pred),
                'classification_report': classification_report(y_test, y_pred)
            }
        return results

    def plot_classifier_comparison(self, results):
        accuracies = [results[clf]['accuracy'] for clf in results]
        clf_names = list(results.keys())
        plt.figure(figsize=(10, 6))
        plt.bar(clf_names, accuracies)
        plt.title('Classifier Performance Comparison')
        plt.xlabel('Classifiers')
        plt.ylabel('Accuracy')
        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()
        plt.show()

    def train_best_classifier(self, classifier_name):
        """
        Train and save the best classifier
        
        Args:
            classifier_name (str): Name of the best classifier
        
        Returns:
            object: Trained classifier
        """
        
        # Select the best classifier
        best_clf = self.classifiers[classifier_name]

        # Save the model and scaler
        with open(os.path.join(util.script_dir, 'classifier.p'), 'wb') as f:
            pickle.dump({'model': best_clf}, f)

        
        return best_clf


def generate_model():
    gesture_classifier = HandGestureClassifier(dataset_path=os.path.join(util.script_dir, util.DATASET_NAME))
    results = gesture_classifier.evaluate_classifiers()
    for name, result in results.items():
        print(f"\n{name} Classifier:")
        print(f"Accuracy: {result['accuracy']:.4f}")
        print("Classification Report:")
        print(result['classification_report'])
    gesture_classifier.plot_classifier_comparison(results)
    best_clf_name = max(results, key=lambda x: results[x]['accuracy'])
    print(f"\nBest Classifier: {best_clf_name}")
    gesture_classifier.train_best_classifier(best_clf_name)
    print("Best classifier trained and saved.")

if __name__ == '__main__':
    generate_model()