from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.metrics import classification_report

def calculate_stats(y_test,y_pred):
    true_positive=None
    true_negative=None
    false_positive=None
    false_negative=None
    
    print('\nClassification Report\n')
    clasifi_report=classification_report(y_test, y_pred, target_names=['neutral', 'sadness', 'joy','anger','tenderness'])
    print(clasifi_report)
    return clasifi_report

def calculate_acc(true_positive,true_negative,false_positive,false_negative):
    return (true_positive+true_negative)/(true_positive+true_negative+false_negative+false_positive)
def calculate_precsion(true_positive,true_negative,false_positive,false_negative):
    return true_positive/(true_positive+true_negative)