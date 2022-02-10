import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn import metrics
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import confusion_matrix
from gensim.test.utils import datapath,get_tmpfile
from gensim.models import KeyedVectors
from gensim.scripts.glove2word2vec import glove2word2vec
from gensim.parsing.preprocessing import remove_stopwords
from nltk.corpus import stopwords
import nltk
from nltk.tokenize import RegexpTokenizer
from sklearn.decomposition import PCA
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
import torchinfo
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from torch.utils.data import random_split
# claims=pd.read_csv('data/claims.csv')
# claims['svc_dt']=claims['svc_dt'].astype('datetime64')
# tokenizer = RegexpTokenizer(r'\w+')
# stop = stopwords.words('english')
# glove_wiki = KeyedVectors.load_word2vec_format('D:/iqivia/project/data/glove.840B.300d.txt', binary=False, no_header=True)
# claims['processed']=claims['svc_cd'].apply(lambda x: [i for i in tokenizer.tokenize(x.lower()) if i not in stop]).str[1:]
# df=pd.DataFrame(claims.groupby('patient_id')['processed'].agg('sum')).reset_index()
# patient=pd.read_csv('data/patient_reference.csv')
# patient['label']=patient.cohort.apply(lambda x: 1 if x=='positive' else 0)
# a=df.merge(patient,left_on='patient_id', right_on='patient_id',how='left')
# def truncate_pad(line, num_steps, padding_token):
#     if len(line) > num_steps:
#         return line[:num_steps]
#     return line + [padding_token] * (num_steps - len(line))
# padded_services=list(a.processed.apply(lambda x: truncate_pad(x,500,'pad')).values)
#
# data=[]
# for padded_service in padded_services:
#     vec=[]
#     for word in padded_service:
#         try:
#             vec.append(glove_wiki[word])
#         except:
#             vec.append(np.random.rand(300))
#
#     data.append(vec)
# X=np.asarray(data)
# y=a.label.values

# np.save('X.npy', X)
# np.save('y.npy', y)




class LSTMTagger(nn.Module):

    def __init__(self, vocab_size,embedding_dim, hidden_dim,  tagset_size):
        super(LSTMTagger, self).__init__()
        self.hidden_dim = hidden_dim
        self.word_embeddings = nn.Embedding(vocab_size, embedding_dim)
        self.lstm = nn.LSTM(embedding_dim, hidden_dim,2,batch_first=True)
        self.drop=nn.Dropout(p=0.3)
        # The linear layer that maps from hidden state space to tag space
        self.hidden2tag = nn.Linear(hidden_dim, tagset_size)

    def forward(self, sentence):
        lstm_out, _ = self.lstm(sentence)
        linear_in=lstm_out[:,-1,:]
        tag_space=self.hidden2tag(linear_in)
        tag_scores = torch.sigmoid(tag_space)
        return tag_scores


def BCELoss_class_weighted(weights):
    def loss(input, target):
        input = torch.clamp(input,min=1e-7,max=1-1e-7)
        bce = - weights[1] * target * torch.log(input) - (1 - target) * weights[0] * torch.log(1 - input)
        return torch.mean(bce)
    return loss



class CSVDataset(Dataset):
    # load the dataset
    def __init__(self, X,y):
        # store the inputs and outputs
        self.X = X.astype('float32')
        self.y = y.reshape(-1,1).astype('float32')
    # number of rows in the dataset
    def __len__(self):
        return len(self.X)
    # get a row at an index
    def __getitem__(self, idx):
        return [self.X[idx], self.y[idx]]
    def get_splits(self, n_test=0.33):
        # determine sizes
        test_size = round(n_test * len(self.X))
        train_size = len(self.X) - test_size
        # calculate the split
        return random_split(self, [train_size, test_size])
def prepare_data(X,y):
    # load the dataset
    dataset = CSVDataset(X,y)
    # calculate split
    train, test = dataset.get_splits()
    # prepare data loaders
    train_dl = DataLoader(train, batch_size=100, shuffle=True,num_workers=5)
    test_dl = DataLoader(test, batch_size=1024, shuffle=False,num_workers=5)
    return train_dl, test_dl
def train(X,y):
    train_dl, test_dl = prepare_data(X,y)


    for epoch in range(3):  # again, normally you would NOT do 300 epochs, it is toy data
        for i,(sentence,tags) in enumerate(train_dl):
            model.zero_grad()
            tag_scores = model(sentence.to(device))
            loss = loss_function(tag_scores, tags.to(device))
            loss.backward()
            optimizer.step()
    model.eval()
    with torch.no_grad():
        for i,(sentence,tags) in enumerate(train_dl):
            tag_scores = model(sentence.to(device)).to('cpu')
            y_pred = tag_scores.detach().numpy()[:,0]
            y_pred_class=[ 1 if prob>0.5 else 0 for prob in y_pred]
            print('train acc:',metrics.accuracy_score(tags, y_pred_class))
            print('train auc:',metrics.roc_auc_score(tags, y_pred))
            print(confusion_matrix(tags, y_pred_class))
        for i,(sentence,tags) in enumerate(test_dl):
            tag_scores = model(sentence.to(device)).to('cpu')
            y_pred = tag_scores.detach().numpy()[:,0]
            y_pred_class=[ 1 if prob>0.5 else 0 for prob in y_pred]
            print('test acc:',metrics.accuracy_score(tags, y_pred_class))
            print('test auc:',metrics.roc_auc_score(tags, y_pred))
            print(confusion_matrix(tags, y_pred_class))
if __name__ == '__main__':
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = LSTMTagger(500, 300, 126, 1)
    model = nn.DataParallel(model)
    model.to(device)
    loss_function = BCELoss_class_weighted(torch.FloatTensor([1, 5]))
    optimizer = optim.SGD(model.parameters(), lr=0.1)
    X = np.load('data/X.npy')
    y = np.load('data/y.npy')
    train(X,y)
# y_pred = model_class.predict(X_train)
# y_pred_class=[ 1 if prob>0.5 else 0 for prob in y_pred]
# print('train acc:',metrics.accuracy_score(y_train, y_pred_class))
# print('train auc:',metrics.roc_auc_score(y_train, y_pred))
# print(confusion_matrix(y_train, y_pred_class))
#
# y_pred = model_class.predict(X_test)
# y_pred_class=[ 1 if prob>0.5 else 0 for prob in y_pred]
# print('test acc:',metrics.accuracy_score(y_test, y_pred_class))
# print('test auc:',metrics.roc_auc_score(y_test, y_pred))
# print(confusion_matrix(y_test, y_pred_class))

