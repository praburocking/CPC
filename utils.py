from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.metrics import classification_report
import pandas as pd
import os
def calculate_stats(y_target,y_pred,target_names=['neutral', 'sadness', 'joy','anger','tenderness']):
    true_positive=None
    true_negative=None
    false_positive=None
    false_negative=None
    
    print('\nClassification Report\n')
    clasifi_report=classification_report(y_target, y_pred, target_names=target_names,output_dict=True)
    df = pd.DataFrame(clasifi_report).transpose()
    print(df.to_markdown())
    return df

def calculate_acc(true_positive,true_negative,false_positive,false_negative):
    return (true_positive+true_negative)/(true_positive+true_negative+false_negative+false_positive)
def calculate_precsion(true_positive,true_negative,false_positive,false_negative):
    return true_positive/(true_positive+true_negative)

def create_experiment_folder(path):
    if not os.path.exists(path):
        os.makedirs(path)
    