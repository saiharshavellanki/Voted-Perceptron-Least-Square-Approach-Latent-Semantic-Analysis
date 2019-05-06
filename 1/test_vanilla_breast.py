import numpy as np

def preprocessing():
    contents = np.genfromtxt('breast-cancer-wisconsin.data', delimiter=',',missing_values='?',usemask=False)
    contents=contents[~np.isnan(contents).any(axis=1)]
    contents=contents.astype(float)
    return contents

def initialization(contents,attributes_num):
    w=np.random.rand(attributes_num)
    m=len(contents)
    b=np.random.rand(1)
    return [w,m,b]

def vanilla_perceptron(contents,w,m,b,num_epochs,attributes_num):
    for it in range(0,num_epochs):
        for i in range(0,m):
            y=3-contents[i][attributes_num+1]
            x=contents[i][1:attributes_num+1]
            if y*(w.dot(x)+b) <=0:
                b=b+y
                w=w+y*x
    return [w,b]

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
    [w,m,b]=initialization(train_data,attributes_num)
    [w,b]=vanilla_perceptron(train_data,w,m,b,num_epochs,attributes_num)
    ans=0
    for i in range(0,len(test_data)):
        y=3-test_data[i][attributes_num+1]
        x=test_data[i][1:attributes_num+1]
        if y*(w.dot(x)+b)>0 :
            ans+=1
    # print ans,len(test_data)
    cum_accuracy+=(ans*1.0)/(len(test_data))
cum_accuracy=cum_accuracy/k
print cum_accuracy
