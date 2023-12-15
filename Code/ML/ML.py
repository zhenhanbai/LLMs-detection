import jieba
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import SVC
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report, roc_curve, roc_auc_score
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression  
from sklearn.naive_bayes import GaussianNB
import matplotlib.pyplot as plt
import logging

logging.basicConfig(level=logging.DEBUG, 
                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger('train')
file_handler = logging.FileHandler('train.log') 
file_handler.setLevel(logging.INFO)
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
file_handler.setFormatter(formatter)
logger.addHandler(file_handler)


def data_process(file_name):
    data = pd.read_csv(file_name, encoding='utf-8')
    data = data[['answer', 'label']]
    data['answer'] = data['answer'].apply(lambda x: ' '.join(jieba.cut(x)))
    return data['answer'], data['label']


def train_model(train_file, test_file, sent_text_file):

    X_train, y_train = data_process(train_file)
    X_test, y_test = data_process(test_file)
    X_sent_test, y_sent_test = data_process(sent_text_file)
    X_concat_test, y_concat_test = pd.concat([X_test, X_sent_test], axis=0), pd.concat([y_test, y_sent_test], axis=0)

    tfidf_vectorizer = TfidfVectorizer()
    X_train = tfidf_vectorizer.fit_transform(X_train)
    X_test = tfidf_vectorizer.transform(X_test)
    X_sent_test = tfidf_vectorizer.transform(X_sent_test)
    X_concat_test = tfidf_vectorizer.transform(X_concat_test)


    classifiers = [
        LogisticRegression(C=62.10, penalty='l2',solver='liblinear',max_iter=1000),
        GaussianNB(),
        KNeighborsClassifier(n_neighbors=96),
        DecisionTreeClassifier(),
        SVC(kernel='linear', C=1, probability=True),
        RandomForestClassifier(),
        GradientBoostingClassifier(),
        XGBClassifier(),
        LGBMClassifier(),
        CatBoostClassifier(logging_level='Silent')
    ]

    roc_curves = {}
    auc_scores = []
    stats = []

    test_ood = True
    if test_ood:
        X_test, y_test = X_sent_test, y_sent_test

    for clf in classifiers:
        model_name = clf.__class__.__name__
        logger.info("----------------------\n %s", model_name)
        if isinstance(clf, GaussianNB):
            clf.fit(X_train.toarray(), y_train)
            y_pred = clf.predict(X_test.toarray())
        else:
            if hasattr(clf, "predict_proba"):
                clf.fit(X_train, y_train)
                y_pred = clf.predict_proba(X_test)[:, 1]
            else:
                y_pred = clf.predict(X_test)

        y_pred = [1 if score > 0.5 else 0 for score in y_pred]
        clf_report = classification_report(y_test, y_pred, target_names=['human','chatgpt'], digits=4, output_dict=True)
        fpr, tpr, thresholds = roc_curve(y_test, y_pred)
        roc_curves[model_name] = (fpr, tpr)
        auc = roc_auc_score(y_test, y_pred)
        auc_scores.append((model_name, auc))
        logger.info("----------------------\n %s", clf_report)
        
        stats.append({
            'model_name': model_name,
            'accuracy': clf_report['accuracy'],
            'weighted_avg_precision': clf_report['weighted avg']['precision'],
            'weighted_avg_recall': clf_report['weighted avg']['recall'],
            'weighted_avg_f1-score': clf_report['weighted avg']['f1-score'],
            'chatgpt_precision': clf_report['chatgpt']['precision'],
            'chatgpt_recall': clf_report['chatgpt']['recall'],
            'chatgpt_f1-score': clf_report['chatgpt']['f1-score'],
            'human_precision': clf_report['human']['precision'],
            'human_recall': clf_report['human']['recall'],
            'human_f1-score': clf_report['human']['f1-score']
        })

    df_stats = pd.DataFrame(data=stats)
    df_stats = df_stats.round(4)
    df_stats.to_csv(f"../../Result/ML/result_{test_ood}.csv", index=None)

    plt.figure(figsize=(10, 6), dpi=200)
    for i, (model_name, (fpr, tpr)) in enumerate(roc_curves.items()):
        auc_value = auc_scores[i][1]
        plt.plot(fpr, tpr, label=f'{model_name} (AUC = {auc_value:.2f})')

    plt.plot([0, 1], [0, 1], linestyle='--', color='gray', label='Random Guess')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curve')
    plt.legend()
    plt.savefig(f"../../Result/ML/result_{test_ood}.png")

if __name__ == "__main__":
    train_model(train_file="../../data/zh_doc_train.csv", test_file="../../data/zh_doc_test.csv", sent_text_file="../../data/shuffled_zh_sent_test.csv")