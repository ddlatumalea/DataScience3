"""Model metrics"""
from sklearn.metrics import classification_report, confusion_matrix

def get_evaluation(y_test, y_pred):
    """Prints the confusion matrix and classification report with individual class performance"""
    print(confusion_matrix(y_test, y_pred))
    print(classification_report(y_test, y_pred))
