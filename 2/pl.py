import numpy as np
import matplotlib.pyplot as plt

# data1=np.array([[3,3],[3,0],[2,1],[0,2]]).astype(float)
# data2=np.array([[-1,1],[0,0],[-1,-1],[1,0]]).astype(float)
data1=np.array([[3,3],[3,0],[2,1],[0,1.5]]).astype(float)
data2=np.array([[-1,1],[0,0],[-1,-1],[1,0]]).astype(float)


# w=np.array([ 0.375,0.33423913,-0.57880435])
w=np.array([0.37779521,0.30207925,-0.53825029])

a = -w[0] / w[1]
xx = np.linspace(-2, 4, 20)
yy = a * xx - (w[2]) / w[1]

plt.plot(xx, yy, c='black', ls='dashed', label='lsa classifier')

c1x = [data1[i][0] for i in range(len(data1))]
c1y = [data1[i][1] for i in range(len(data1))]
c2x = [data2[i][0] for i in range(len(data2))]
c2y = [data2[i][1] for i in range(len(data2))]

plt.scatter(c1x, c1y, c='blue')
plt.scatter(c2x, c2y, c='green')
# w=np.array([0.74651327,0.66537052,-1.03862716])
w=np.array([0.78102704,0.62449721,-0.85888643])
a = -w[0] / w[1]
xx = np.linspace(-2, 4, 20)
yy = a * xx - (w[2]) / w[1]

plt.plot(xx, yy, 'r--', label='fischer lda classifier')
plt.legend()

plt.show()
