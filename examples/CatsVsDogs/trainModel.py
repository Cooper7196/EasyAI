from imageRecognition import *

model = train("C:\\Users\\cbros\\PycharmProjects\\EasyAI\\examples\\CatsVsDogs\\dataset\\training_set")
result = classify(model, "C:\\Users\\cbros\\PycharmProjects\\EasyAI\\examples\\CatsVsDogs\\dataset\\test_set\\dogs\\dog.4001.jpg")
print(result)
save(model, "model")