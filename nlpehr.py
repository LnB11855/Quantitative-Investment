"""
idea： 加权重（时间远近和序列选择比例）生成不同序列
transformers
Clinical bert

"""


import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression,LinearRegression,Ridge
from sklearn.ensemble import RandomForestRegressor
from sklearn import metrics
from numpy import loadtxt

import matplotlib.pyplot as plt
import numpy as np
from sklearn.preprocessing import OneHotEncoder,MinMaxScaler
from sklearn.svm import SVC
import tensorflow.keras.backend as K
from sklearn.feature_selection import chi2
from statsmodels.stats.outliers_influence import variance_inflation_factor
from sklearn.metrics import confusion_matrix

from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.layers import Dense,Flatten,LSTM,GRU,concatenate
from tensorflow.keras.layers import Input,BatchNormalization,Dropout
from tensorflow.keras.models import Model
from tensorflow.compat.v1 import ConfigProto
from tensorflow.compat.v1 import InteractiveSession
plt.figure()
claims=pd.read_csv('data/claims.csv')
claims['svc_dt']=claims['svc_dt'].astype('datetime64')
claims.groupby([claims["svc_dt"].dt.year, claims["svc_dt"].dt.month]).count().plot(kind="bar")
plt.show()
plt.figure()
claims.svc_dt.hist()
plt.title('Hist of service date')
plt.show()

print('Number of patient: %d'%claims.patient_id.nunique())
print('Number of doctor : %d'%claims.provider_id.nunique())
print('Number of svc_typ : %d'%claims.svc_typ.nunique())
print('Number of svc_cd : %d'%claims.svc_cd.nunique())
print('Number of svc_index : %d'%claims.svc_idx.nunique())

service=pd.read_csv('data/service_reference.csv')
service.group_1.nunique()/service.group_1.count()
service.group_2.nunique()/service.group_2.count()
service[service.group_1==service.group_2].count()/service.count()

plt.figure()
a=claims.groupby(['patient_id']).svc_dt.agg(np.ptp).astype('timedelta64[D]').values
plt.hist(a,50)
plt.title('Hist of patient service date')
plt.show()

plt.figure()
a=claims.groupby(['provider_id']).svc_dt.agg(np.ptp).astype('timedelta64[D]').values
plt.hist(a,50)
plt.title('Hist of doctor service date')
plt.show()


from gensim.test.utils import datapath,get_tmpfile
from gensim.models import KeyedVectors
from gensim.scripts.glove2word2vec import glove2word2vec
from gensim.parsing.preprocessing import remove_stopwords
from nltk.corpus import stopwords
import nltk
from nltk.tokenize import RegexpTokenizer
from sklearn.decomposition import PCA


tokenizer = RegexpTokenizer(r'\w+')
stop = stopwords.words('english')
service['processed']=service['usc_desc'].apply(lambda x: [i for i in tokenizer.tokenize(x.lower()) if i not in stop])

from gensim.models import Word2Vec
model_wv=Word2Vec(list(service.processed.values),min_count=1)
X=model_wv.wv.vectors
pca=PCA(n_components=2)
result=pca.fit_transform(X)
plt.figure(figsize=(40,40))
plt.scatter(result[:,0],result[:,1])
for i,word in enumerate(model_wv.wv.key_to_index.keys()):
    plt.annotate(word,xy=(result[i, 0], result[i, 1]),fontsize=10)
plt.savefig('word')
plt.show()

# glove_file=datapath('C:/Users/lbnfo/Desktop/iqivia/project/data/glove.840B.300d.txt')
# tmp_file=get_tmpfile('data/word2vec.txt')
# glove2word2vec(glove_file,tmp_file)
glove_wiki = KeyedVectors.load_word2vec_format('D:/iqivia/project/data/glove.840B.300d.txt', binary=False, no_header=True)
glove_wiki.most_similar(positive=['woman', 'king'], negative=['man'], topn=1)
claims['processed']=claims['svc_cd'].apply(lambda x: [i for i in tokenizer.tokenize(x.lower()) if i not in stop]+['sep']).str[1:]
claims.sort_values(by='svc_dt', inplace=True)

df=pd.DataFrame(claims.groupby('patient_id')['processed'].agg('sum')).reset_index()

patient=pd.read_csv('data/patient_reference.csv')
patient['label']=patient.cohort.apply(lambda x: 1 if x=='positive' else 0)
a=df.merge(patient,left_on='patient_id', right_on='patient_id',how='left')
plt.hist(a.processed.str.len().values,bins=50)
plt.show()

def truncate_pad(line, num_steps, padding_token):
    if len(line) > num_steps:
        return line[:num_steps]
    return line + [padding_token] * (num_steps - len(line))
padded_services=list(a.processed.apply(lambda x: truncate_pad(x,500,'pad')).values)
data=[]
unrecognize=set()
recognize=set()
for padded_service in padded_services:
    vec=[]
    for word in padded_service:
        try:
            vec.append(glove_wiki[word])
            recognize.add(word)
        except:
            # vec.append(np.random.rand(300))
            unrecognize.add(word)

    data.append(vec)
X=np.asarray(data)
y=a.label.values

X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y,shuffle=True,test_size=0.2,
                                                    random_state=42)

def _construct_model( sentence_lenth=500, word_size=300,lstm_size=64):
    # Create context encoder
    weather_input = Input(shape=(sentence_lenth, word_size),
                         name='weather_input')
    x = LSTM(units=lstm_size,
             return_sequences=True,
             name='x')(weather_input)
    x1 = LSTM(units=lstm_size,
             return_sequences=False,
             name='x1')(x)
    fc1 = Dropout(0.2)(Dense(units=100,
                          activation='relu',
                          name='fc1')(x1))

    fc2 = Dropout(0.3)(Dense(units=100,
                          activation='relu',
                          name='fc2')(fc1))
    model_output=Dense(units=1,
                       activation='sigmoid',
                          name='y2')(fc2)

    model = Model(inputs=weather_input,
                  outputs=model_output)
    opt = Adam(learning_rate=0.001)
    model.compile(loss="binary_crossentropy", optimizer=opt)

    return model
model_class=_construct_model()
model_class.summary()

#%%

early_stopping = EarlyStopping(patience=100,
                               restore_best_weights=True)


history = model_class.fit(X_train, y_train,
                             epochs=10000,
                             batch_size=100,
                             validation_split=0.2,
                            shuffle = True,
                            class_weight={0:1.,1:6.5},
                             callbacks=[early_stopping],
                             verbose=1)


y_pred = model_class.predict(X_train)
y_pred_class=[ 1 if prob>0.5 else 0 for prob in y_pred]
print('train acc:',metrics.accuracy_score(y_train, y_pred_class))
print('train auc:',metrics.roc_auc_score(y_train, y_pred))
print(confusion_matrix(y_train, y_pred_class))

y_pred = model_class.predict(X_test)
y_pred_class=[ 1 if prob>0.5 else 0 for prob in y_pred]
print('test acc:',metrics.accuracy_score(y_test, y_pred_class))
print('test auc:',metrics.roc_auc_score(y_test, y_pred))
print(confusion_matrix(y_test, y_pred_class))
