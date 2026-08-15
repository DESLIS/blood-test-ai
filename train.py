from model import NeuralNetwork

model = NeuralNetwork()

for epoch in range(100):
    model.train_one_epoch()

print(model.loss)