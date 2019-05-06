import matplotlib.pyplot as pl
# x=[5,10,15,20,25]
# y=[0.897849462366,0.954301075269,0.948924731183,0.948924731183,0.948924731183]

##no of epochs =5
x=[0.95,0.8,0.7,0.6,0.5]
#multiclass_perceptron
# y=[0.8978,0.879,0.93,0.9516,0.94086]
#cosine similarity
y=[0.9139,0.9166,0.9139,0.9112,0.9166]
pl.xlabel("Threshold")
pl.ylabel("Accuracy")
pl.title("cosine similarity")
pl.plot(x,y,'r')
pl.show()
