import numpy as np
import sys

def get_means(data1,data2):
    sum1=np.sum(data1,axis=0)
    l1=len(data1)
    sum2=np.sum(data2,axis=0)
    l2=len(data2)
    mean1=sum1/l1
    mean2=sum2/l2
    return [mean1,mean2]

def covariance(data1,mean1,data2,mean2):
    new_data1=np.empty([len(data1),len(data1[0])])
    for i in range(0,len(data1)):
        new_data1[i]=data1[i]-mean1
    new_data1_t=new_data1.transpose()
    cov1=np.matmul(new_data1_t,new_data1)

    new_data2=np.empty([len(data2),len(data2[0])])
    for i in range(0,len(data2)):
        new_data2[i]=data2[i]-mean2
    new_data2_t=new_data2.transpose()
    cov2=np.matmul(new_data2_t,new_data2)

    cov=cov1+cov2
    return cov

p=sys.argv[1]
if int(p)==1:
	data1=np.array([[3,3],[3,0],[2,1],[0,2]]).astype(float)
	data2=np.array([[-1,1],[0,0],[-1,-1],[1,0]]).astype(float)
else:
	data1=np.array([[3,3],[3,0],[2,1],[0,1.5]]).astype(float)
	data2=np.array([[-1,1],[0,0],[-1,-1],[1,0]]).astype(float)

[mean1,mean2]=get_means(data1,data2)
cov=covariance(data1,mean1,data2,mean2)
Inv_cov=np.linalg.inv(np.matrix(cov))
mu_diff=(mean1-mean2).transpose()
W=np.matmul(Inv_cov,mu_diff)
W=W/np.linalg.norm(W)

min_b=-sys.maxint-1
max_b=sys.maxint
for i in range(0,len(data1)):
    min_b=max(min_b,-1*(W.dot(data1[i])))
for i in range(0,len(data2)):
    max_b=min(max_b,-1*(W.dot(data2[i])))
b=(min_b+max_b)/2
b=np.array(np.array(b))
W=np.concatenate((W,b),axis=1)
print W
