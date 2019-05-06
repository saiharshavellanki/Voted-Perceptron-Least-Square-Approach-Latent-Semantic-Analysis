import numpy as np

def preprocessing():
    contents = np.genfromtxt('breast-cancer-wisconsin.data', delimiter=',',missing_values='?',usemask=False)
    contents=contents[~np.isnan(contents).any(axis=1)]
    contents=contents.astype(float)
    return contents

def initialization(contents,attributes_num):
    w=np.empty((0,attributes_num),dtype=float)
    w=np.append(w,np.zeros((1,attributes_num)),axis=0)
    w=np.append(w,np.zeros((1,attributes_num)),axis=0)
    n=1
    m=len(contents)
    b=np.array([],dtype=float)
    b=np.append(b,[0,0])
    c=np.array([],dtype=float)
    c=np.append(c,[1,1])
    return [w,n,m,b,c]

def voted_perceptron(contents,w,n,m,b,c,num_epochs,attributes_num):
    for it in range(0,num_epochs):
        for i in range(0,m):
            y=3-contents[i][attributes_num+1]
            x=contents[i][1:attributes_num+1]
            if y*(w[n].dot(x)+b[n]) <=0:
                n=n+1
                w=np.append(w,np.array([(w[n-1]+y*x).astype(float)]),axis=0)
                b=np.append(b,b[n-1]+y)
                c=np.append(c,1)
            else:
                c[n]=c[n]+1
    w=w.astype(float)
    b=b.astype(float)
    c=c.astype(float)
    return [w,b,c,n]

contents=preprocessing()
attributes_num=len(contents[0])-2
num_epochs=int(raw_input())
k=int(raw_input())
l=len(contents)
p=l/k
cum_accuracy=0
for ind in range(1,k+1):
    #test set-> [(ind-1)*p , (ind)*p)
    if ind<k:
        end=ind*p
    else:
        end=l
    test_data=contents[(ind-1)*p:end]
    train_data=np.append(contents[0:(ind-1)*p],contents[end:l],axis=0)
    [w,n,m,b,c]=initialization(train_data,attributes_num)
    [w,b,c,n]=voted_perceptron(train_data,w,n,m,b,c,num_epochs,attributes_num)
    co=0
    for i in range(0,len(test_data)):
        ans=0
        for j in range(1,n+1):
            x=test_data[i][1:attributes_num+1]
            if w[j].dot(x)+b[j]>0:
                ans+=c[j]
            else:
                ans-=c[j]
        if ans>0:
            y=1
        else:
            y=-1
        if test_data[i][attributes_num+1]==2.0 and y==1.0:
            co+=1
        if test_data[i][attributes_num+1]==4.0 and y==-1.0:
            co+=1
    cum_accuracy+=(co*1.0)/len(test_data)
cum_accuracy=cum_accuracy/k
print cum_accuracy
