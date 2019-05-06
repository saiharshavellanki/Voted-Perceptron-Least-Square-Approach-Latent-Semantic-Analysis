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

def multiclass_perceptron(new_data,k,Y):
    print "give no of epochs"
    num_epochs=int(raw_input())
    w=np.zeros((5,k+1),dtype="float")
    for i in range(0,num_epochs):
        for j in range(0,len(new_data)):
            x=np.array(new_data[j].astype(float))
            y=int(Y[j])
            maxi=-sys.maxint
            pos=0
            for p in range(0,5):
                dist=np.dot(w[p],x)
                if dist>= maxi:
                    pos=p
                    maxi=dist
            if not (y==pos):
                w[y]=w[y]+x
                w[pos]=w[pos]-x
    return w

def predict_test(X,Y):
    passed=0
    for i in range(0,len(X)):
        x=np.array(X[i].astype(float))
        y=int(Y[i])
        maxi=-sys.maxint
        pos=0
        for p in range(0,5):
            dist=np.dot(w[p],x)
            if dist>= maxi:
                pos=p
                maxi=dist
        if pos==y:
            passed+=1
    return (passed*1.0)/(len(X)*1.0)

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
w=multiclass_perceptron(X_train,k,Y_train)

file_names=get_filenames(test_path)
total_no_of_docs=no_of_test_docs(file_names)
[X_test,Y_test,n]=get_inputs(test_path,file_names,word_index,c,stop_words,total_no_of_docs)
X_test=transform_X(X_test)
X_test=np.matmul(X_test,v)

# md = pc.Perceptron()
# md = md.fit(X_train,Y_train)
# sc = md.score(X_test,Y_test)

accuracy=predict_test(X_test,Y_test)
print accuracy
