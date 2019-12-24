import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import SVC

data = pd.read_table('spam/SMSSpamCollection', sep='\t', names=['label', 'message'])
data['label'] = data.label.map({'ham': 0, 'spam': 1})  # 字符串转为数值

# 划分训练集、测试集
train_msg, test_msg, train_label, test_label = train_test_split(data['message'], data['label'], random_state=0,
                                                                test_size=0.2, shuffle=True)

# 得到训练集和测试集的tfidf矩阵
tfidf_vec = TfidfVectorizer(binary=False, decode_error='ignore', stop_words='english')
train_data = tfidf_vec.fit_transform(train_msg)
test_data = tfidf_vec.transform(test_msg)

# 朴素贝叶斯方法
naive_bayes = MultinomialNB(alpha=1.0, class_prior=None, fit_prior=True)
naive_bayes.fit(train_data, train_label)
predictions = naive_bayes.predict(test_data)
# print('Accuracy score: ', format(accuracy_score(test_label, predictions)))
# print('Precision score: ', format(precision_score(test_label, predictions)))
# print('Recall score: ', format(recall_score(test_label, predictions)))
# print('F1 score: ', format(f1_score(test_label, predictions)))
print(classification_report(test_label, predictions, target_names=['ham', 'spam'], digits=4))

# SVM方法
svc = SVC(kernel='linear', class_weight='balanced')
svc.fit(train_data, train_label)
predictions = svc.predict(test_data)
# print('Accuracy score: ', format(accuracy_score(test_label, predictions)))
# print('Precision score: ', format(precision_score(test_label, predictions)))
# print('Recall score: ', format(recall_score(test_label, predictions)))
# print('F1 score: ', format(f1_score(test_label, predictions)))
print(classification_report(test_label, predictions, target_names=['ham', 'spam'], digits=4))

# result
# 词袋模型
# naive_bayes
#               precision    recall  f1-score   support
#
#          ham     0.9886    0.9969    0.9927       955
#         spam     0.9803    0.9313    0.9551       160
# svm
#               precision    recall  f1-score   support
#
#          ham     0.9835    0.9990    0.9912       955
#         spam     0.9931    0.9000    0.9443       160
#

# tf-idf模型
# naive_bayes
#               precision    recall  f1-score   support
#
#          ham     0.9735    1.0000    0.9866       955
#         spam     1.0000    0.8375    0.9116       160
#
# svm
#               precision    recall  f1-score   support
#
#          ham     0.9896    0.9958    0.9927       955
#         spam     0.9740    0.9375    0.9554       160
