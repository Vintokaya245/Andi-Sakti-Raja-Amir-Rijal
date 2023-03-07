import warnings

import numpy as np
import pandas as pd
import torch
import transformers as ppb
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from tqdm.notebook import tqdm_notebook

tqdm_notebook.pandas()

warnings.filterwarnings('ignore')

df = pd.read_csv('https://github.com/clairett/pytorch-sentiment-classification/raw/master/data/SST2/train.tsv',
                 delimiter='\t', header=None)

df = df.dropna(how='all')
print('df', df)
##df.head(2)
X = df[0]
Y = df[1]
encoder = LabelEncoder()
Y = encoder.fit_transform(Y)
print(Y, 'Y')
x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=0.3)

## Pre Processing

class BertTokenizer():
    def __init__(self, text=[]):
        self.text = text
        # For DistilBERT:
        self.model_class, self.tokenizer_class, self.pretrained_weights = (
            ppb.DistilBertModel, ppb.DistilBertTokenizer, 'distilbert-base-uncased')

        # Load pretrained model/tokenizer
        self.tokenizer = self.tokenizer_class.from_pretrained(self.pretrained_weights)
        self.model = self.model_class.from_pretrained(self.pretrained_weights)

    def get(self):
        df = pd.DataFrame(data={"text": self.text})
        tokenized = df["text"].swifter.apply((lambda x: self.tokenizer.encode(x, add_special_tokens=True)))
        max_len = 0
        for i in tokenized.values:
            if len(i) > max_len:
                max_len = len(i)

        padded = np.array([i + [0] * (max_len - len(i)) for i in tokenized.values])

        attention_mask = np.where(padded != 0, 1, 0)
        input_ids = torch.tensor(padded)
        attention_mask = torch.tensor(attention_mask)

        with torch.no_grad():
            last_hidden_states = self.model(input_ids, attention_mask=attention_mask)

        features = last_hidden_states[0][:, 0, :].numpy()
        return features


_instance = BertTokenizer(text="x_train")
tokens = _instance.get()
print(tokens, 'Tokens')

##Model

lr_clf = LogisticRegression()
lr_clf.fit(tokens, y_train)
print(lr_clf, 'lr_clf')

##Test
_instance = BertTokenizer(text="x_test")
tokensTest = _instance.get()
print(tokensTest, 'Token Test')
predicted = lr_clf.predict(tokensTest)
print(predicted, 'Prediksi')
Accuration = np.mean(predicted == y_test)
print(y_test, 'Ytest')
print('Akurasi', Accuration)