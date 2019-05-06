import matplotlib.pyplot as pl

#vanilla
x1=[10,15,20,25,30,35,40,45,50]
y1=[0.86,0.86,0.8714,0.8771,0.8628,0.8686,0.8685,0.8828,0.8714]
pl.xlabel("No of Epochs")
pl.ylabel("Cross validation accuracy")
pl.title("ionosphere data") 
#pl.plot(x1,y1,'r')

#voted
x2=[10,15,20,25,30,35,40,45,50]
y2=[0.8572,0.86,0.8629,0.8658,0.8686,0.8772,0.8772,0.88,0.88]
#pl.plot(x2,y2,'blue', label='ionosphere voted_perceptron')

#pl.show()


pl.figure(1)
pl.plot(x1, y1, 'blue', label='vanilla perceptron ionosphere')
pl.plot(x2, y2, 'brown', label='voted perceptron ionosphere')
pl.legend()
#plt.figure(2)
#plt.plot(epochs, cancer_acc_voted, 'green', label='cancer voted_perceptron')
#plt.plot(epochs, cancer_acc_normal, 'red', label='cancer perceptron')
#plt.legend()
pl.show()
