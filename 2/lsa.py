import numpy as np
import matplotlib.pyplot as pl
import sys

p=sys.argv[1]
if int(p)==1:
	A=np.array([[3,3,1],[3,0,1],[2,1,1],[0,2,1],[-1,1,1],[0,0,1],[-1,-1,1],[1,0,1]])
else:
	A=np.array([[3,3,1],[3,0,1],[2,1,1],[0,1.5,1],[-1,1,1],[0,0,1],[-1,-1,1],[1,0,1]])

Y=[[1],[1],[1],[1],[-1],[-1],[-1],[-1]]
At=A.transpose()
At_A=np.matmul(At,A)
I=np.linalg.inv(np.matrix(At_A))
W=(I*At)*Y
print W.transpose()
