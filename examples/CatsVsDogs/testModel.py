from imageRecognition import *

model = load("model")
result = classify(model, "dataset\\test_set\\dogs\\dog.4001.jpg")
print(result)
