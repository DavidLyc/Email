import pandas as pd
from sklearn.feature_extraction.text import TfidfTransformer, CountVectorizer

data = pd.read_table('spam/SMSSpamCollection', sep='\t', names=['label', 'message'])
data['label'] = data.label.map({'ham': 0, 'spam': 1})   # 字符串转为数值
count = CountVectorizer().fit_transform(data['message'])  # 统计词频
tfidf_matrix = TfidfTransformer().fit_transform(count)  # 得到tfidf矩阵
print(tfidf_matrix)
