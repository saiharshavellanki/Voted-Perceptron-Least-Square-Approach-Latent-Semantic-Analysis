import os
import math
import sys
import numpy as np
from scipy.sparse import csr_matrix
from scipy import spatial
import sklearn.feature_extraction.text as sk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import sklearn.linear_model as pc

reload(sys)
sys.setdefaultencoding('utf8')

def is_ascii(s):
    return all(ord(c) < 128 for c in s)

def get_filenames(input):
    file_names=[]
    for i in range(0,5):
        file_names.append(list(os.listdir(input+"/"+str(i))))
    return file_names

def preprocessing(input,total_no_of_docs,word_index,c):
    stop_words = set(stopwords.words('english'))
    for i in range(0,5):
        for j in range(0,len(file_names[i])):
            total_no_of_docs+=1
            f=open(input+"/"+str(i)+"/"+file_names[i][j],"r")
            data=f.read()
            f.close()
            data=data.split("\n")
            for line in data:
                line = word_tokenize(line)
                for word in line:
                    word=word.lower()
                    if word not in stop_words and word not in word_index and len(word)>1:
                        word_index[word]=c
                        c+=1
    return [total_no_of_docs,word_index,c,stop_words]

def get_inputs(input,file_names,word_index,c,stop_words,total_no_of_docs):
    X=np.empty([total_no_of_docs,c],dtype=float)
    Y=np.empty(total_no_of_docs,dtype=float)
    n=0
    for i in range(0,5):
        for j in range(0,len(file_names[i])):
            doc_vector=np.zeros(c,dtype=float)
            f=open(input+"/"+str(i)+"/"+file_names[i][j],"r")
            data=f.read()
            f.close()
            data=data.split("\n")
            for line in data:
                line = word_tokenize(line)
                for word in line:
                    word=word.lower()
                    if word not in stop_words  and word in word_index and len(word)>1 and word:
                        doc_vector[word_index[word]]+=1
            X[n]=doc_vector
            Y[n]=i
            n=n+1

    return [X,Y,n]

def transform_X(X):
    model=sk.TfidfTransformer(use_idf=True).fit(X)
    X=model.transform(X)
    X=csr_matrix(X).toarray()
    return X

def transform_matrix(X):
    u,s,v=np.linalg.svd(X,full_matrices=0)
    total_sum=0.0
    for i in range(0,len(s)):
        total_sum+=s[i]
    curr_sum=0
    for i in range(0,len(s)):
        curr_sum+=s[i]
        if curr_sum>=0.95*total_sum:
            break
    k=i
    v=v[0:k+1,:]
    v=v.transpose()
    del(u,s)
    return [v,k]

def no_of_test_docs(file_names):
    total_no_of_docs=0
    for i in range(0,5):
        for j in range(0,len(file_names[i])):
            total_no_of_docs+=1
    return total_no_of_docs

train_path=sys.argv[1]
test_path=sys.argv[2]

file_names=get_filenames(train_path)
[total_no_of_docs,word_index,c,stop_words]=preprocessing(train_path,0,{},0)
[X_train,Y_train,n]=get_inputs(train_path,file_names,word_index,c,stop_words,total_no_of_docs)
X_train=transform_X(X_train)
[v,k]=transform_matrix(X_train)
X_train=np.matmul(X_train,v)

file_names=get_filenames(test_path)
total_no_of_docs=no_of_test_docs(file_names)
[X_test,Y_test,n]=get_inputs(test_path,file_names,word_index,c,stop_words,total_no_of_docs)
X_test=transform_X(X_test)
X_test=np.matmul(X_test,v)

ans=0
#cosine similarity
for i in range(0,len(X_test)):
    similarity=np.zeros((len(X_train),2),dtype="float")
    for j in range(0,len(X_train)):
        similarity[j][0]=1-spatial.distance.cosine(X_test[i],X_train[j])
        similarity[j][1]=Y_train[j]
    # print similarity
    similarity = similarity[similarity[:,0].argsort()]
    similarity=np.flip(similarity,0)
    # similarity=np.sort(similarity.view('i8,i8'), order=['f1'], axis=0)
    labels=[0,0,0,0,0]
    for j in range(0,10):
        labels[int(similarity[j][1])]+=1

    max_val=0
    max_label=0
    for j in range(0,5):
        if labels[j]>max_val:
            max_val=labels[j]
            max_label=j
    # print "prediced label:",max_label,"original label:",Y_test[i]
    if max_label==int(Y_test[i]):
        ans+=1
print ans
print len(X_test)
print (ans*1.0)/(len(X_test)*1.0)
