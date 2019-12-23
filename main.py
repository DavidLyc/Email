import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB

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
print('Accuracy score: ', format(accuracy_score(test_label, predictions)))
print('Precision score: ', format(precision_score(test_label, predictions)))
print('Recall score: ', format(recall_score(test_label, predictions)))
print('F1 score: ', format(f1_score(test_label, predictions)))

# result
# 词袋模型
# Accuracy score:  0.9874439461883409
# Precision score:  0.9802631578947368
# Recall score:  0.93125
# F1 score:  0.9551282051282052
#
# tfidf
# Accuracy score:  0.9766816143497757
# Precision score:  1.0
# Recall score:  0.8375
# F1 score:  0.9115646258503401
