import matplotlib.pylab as plt
import tensorflow as tf
import tensorflow_hub as hub
from tensorflow.keras import layers
import numpy as np
import pickle
import cv2


def train(trainingDirectory, imageSize=(224, 224), epochs=1):
    image_generator = tf.keras.preprocessing.image.ImageDataGenerator(rescale=1 / 255)
    image_data = image_generator.flow_from_directory(trainingDirectory,
                                                     target_size=imageSize)

    # "C:\\Users\\cbros\\Downloads\\dataset\\training_set"

    for image_batch, label_batch in image_data:
        break

    feature_extractor_url = "https://tfhub.dev/google/tf2-preview/mobilenet_v2/feature_vector/4"

    feature_extractor_layer = hub.KerasLayer(feature_extractor_url,
                                             input_shape=(224, 224, 3))
    feature_batch = feature_extractor_layer(image_batch)
    feature_extractor_layer.trainable = False

    model = tf.keras.Sequential([
        feature_extractor_layer,
        layers.Dense(image_data.num_classes)
    ])

    predictions = model(image_batch)

    model.compile(
        optimizer=tf.keras.optimizers.Adam(),
        loss=tf.keras.losses.CategoricalCrossentropy(from_logits=True),
        metrics=['acc'])

    steps_per_epoch = np.ceil(image_data.samples / image_data.batch_size)

    history = model.fit(image_data, epochs=epochs, steps_per_epoch=steps_per_epoch)

    class_names = sorted(image_data.class_indices.items(), key=lambda pair: pair[1])
    class_names = np.array([key.title() for key, value in class_names])

    return (model, class_names, imageSize)


# predicted_batch = model.predict(image_batch)
# predicted_id = np.argmax(predicted_batch, axis=-1)
# predicted_label_batch = class_names[predicted_id]
#
# label_id = np.argmax(label_batch, axis=-1)
#
# plt.figure(figsize=(10, 9))
# plt.subplots_adjust(hspace=0.5)
# for n in range(30):
#     plt.subplot(6, 5, n + 1)
#     plt.imshow(image_batch[n])
#     color = "green" if predicted_id[n] == label_id[n] else "red"
#     plt.title(predicted_label_batch[n].title(), color=color)
#     plt.axis('off')
# _ = plt.suptitle("Model predictions (green: correct, red: incorrect)")
# plt.show()
#
#
# model.save("", save_format='tf')

def save(model, path):
    model[0].save(path, save_format='tf')
    classes = open(path + "\\classes", "wb")
    size = open(path + "\\size", "wb")
    pickle.dump(model[1], classes)
    pickle.dump(model[2], size)


def load(path):
    classes = open(path + "\\classes", 'rb')
    size = open(path + "\\size", "rb")
    return tf.keras.models.load_model(path), pickle.load(classes), pickle.load(size)


def classify(modelData, imagePath):
    model = modelData[0]

    x = cv2.imread(imagePath)
    x = cv2.resize(x, (modelData[2]))
    x = x.reshape((1,) + x.shape)

    class_names = modelData[1]

    predicted_batch = model.predict(x)
    predicted_id = np.argmax(predicted_batch, axis=-1)
    predicted_label_batch = class_names[predicted_id]

    return predicted_label_batch[0]


# model = train("C:\\Users\\cbros\\Downloads\\dataset\\training_set")
# model = train("C:\\Users\\cbros\\Downloads\\seg_train")
# save(model, "test")
model = load("catsVsDogs")
print(classify(model, "C:\\Users\\cbros\\Downloads\\samoyed_puppy_dog_pictures.jpg"))
