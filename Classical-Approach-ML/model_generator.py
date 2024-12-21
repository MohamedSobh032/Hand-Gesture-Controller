import pickle
import numpy as np
import os
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
import util

class HandGestureClassifier:
    def __init__(self, dataset_path: str):
        '''
        Initialize the HandGestureClassifier class by
            - reading the dataset from the given path
            - defining the classifiers to be comapred
        '''

        # Load the dataset
        with open(dataset_path, 'rb') as f:
            dataset = pickle.load(f)
        self.features = dataset['features']
        self.labels = dataset['labels']
        
        # Define the classifiers
        self.classifiers = {
            'KNN': KNeighborsClassifier(n_neighbors=7),
            'Decision Tree': DecisionTreeClassifier(),
            'Random Forest': RandomForestClassifier()
        }

    def evaluate_classifiers(self, test_size: int = 0.2, random_state: int = 42):
        '''
        Evaluate the classifiers by
            - splitting the dataset into training and testing sets
            - training each classifier on the training set
            - evaluating each classifier on the testing set
        '''

        # Split the dataset
        X_train, X_test, y_train, y_test = train_test_split(self.features, self.labels, test_size=test_size, random_state=random_state)
        results = {}

        # For each classifier, train and evaluate
        for name, clf in self.classifiers.items():

            # train the classifier
            clf.fit(X_train, y_train)
            y_pred = clf.predict(X_test)

            # evaluate the classifier
            results[name] = {
                'accuracy': accuracy_score(y_test, y_pred),
                'classification_report': classification_report(y_test, y_pred)
            }

        # return the results
        return results

    def plot_classifier_comparison(self, results):
        '''
        Plot the comparison of the classifiers based on the accuracy
        '''
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

    def train_best_classifier(self, classifier_name: str) -> None:
        """
        Saves the best classifier to a file
        
        Args:
            classifier_name (str): Name of the best classifier
        """
        
        # Select the best classifier
        best_clf = self.classifiers[classifier_name]

        # Save the model and scaler
        with open(os.path.join(util.script_dir, util.MODEL_NAME), 'wb') as f:
            pickle.dump({'model': best_clf}, f)


def generate_model():
    '''
    Generate the model by training the best classifier
    '''

    # Initialize the HandGestureClassifier
    gesture_classifier = HandGestureClassifier(dataset_path=os.path.join(util.script_dir, util.DATASET_NAME))

    # Evaluates the classifiers
    results = gesture_classifier.evaluate_classifiers()

    # Prints the results
    for name, result in results.items():
        print(f"\n{name} Classifier:")
        print(f"Accuracy: {result['accuracy']:.4f}")
        print("Classification Report:")
        print(result['classification_report'])
    
    # Plot for visualization comparison
    gesture_classifier.plot_classifier_comparison(results)

    # Evaluate the best model based on accuracy
    best_clf_name = max(results, key=lambda x: results[x]['accuracy'])
    print(f"\nBest Classifier: {best_clf_name}")
    gesture_classifier.train_best_classifier(best_clf_name)
    print("Best classifier trained and saved.")

if __name__ == '__main__':
    generate_model()