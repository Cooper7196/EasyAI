from imageRecognition import *

model = load("catsVsDogs")
test = classify(model, "C:\\Users\\cbros\\Downloads\\samoyed_puppy_dog_pictures.jpg")
# print()
