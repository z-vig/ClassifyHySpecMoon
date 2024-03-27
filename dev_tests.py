from classifyhyspecmoon.neuralnetdata import NeuralNetData
import matplotlib.pyplot as plt

mydata = NeuralNetData('C:/Users/zvig/.julia/dev/JENVI.jl/Data/targeted.hdf5')

print(mydata.raw_spec.shape)

plt.plot(mydata.raw_spec[:,10,10])
plt.show()