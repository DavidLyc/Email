from sklearn.feature_extraction.text import TfidfTransformer, CountVectorizer
import pandas as pd

data = pd.read_table('spam/SMSSpamCollection', sep='\t', names=['label', 'message'])
data['label'] = data.label.map({'ham': 0, 'spam': 1})
label_list = []
msg_list = []
for label in data['label']:
    label_list.append(label)
for msg in data['message']:
    msg_list.append(msg)
print(len(msg_list))
