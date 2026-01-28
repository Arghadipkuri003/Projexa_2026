import matplotlib.pyplot as plt

epochs = [1,2,3,4,5,6,7,8,9,10]
accuracy = [60,65,70,78,85,88,91,93,94,95]

plt.plot(epochs, accuracy)
plt.xlabel("Epochs")
plt.ylabel("Accuracy (%)")
plt.title("Model Accuracy vs Epochs")
plt.savefig("static/charts/accuracy.png")
plt.close()
