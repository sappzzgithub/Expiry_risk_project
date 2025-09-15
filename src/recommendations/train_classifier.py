"""
Action Classifier
-----------------
- Trains RandomForestClassifier to predict actions (including Relocate)
"""

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder

def train_classifier(X, y):
    le = LabelEncoder()
    y_enc = le.fit_transform(y)

    X_train, X_test, y_train, y_test = train_test_split(X, y_enc, test_size=0.2, random_state=42)
    clf = RandomForestClassifier(random_state=42, class_weight="balanced")
    clf.fit(X_train, y_train)
    return clf, le