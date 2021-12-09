import numpy as np
from tensorflow.keras.applications import EfficientNetB0
from sam import SAM
import tensorflow_datasets as tfds
from matplotlib import pyplot as plt
import tensorflow as tf

np.random.seed(123)

(dogs_train,dogs_test), dogs_info = tfds.load('stanford_dogs',split=['train', 'test'],
                                                with_info=True,as_supervised= True,download=True)

dogs_classes = dogs_info.features["label"].num_classes

(food_train, food_test), food_info = tfds.load('food101:2.1.0', split=['train', 'validation'],
                                               with_info=True,as_supervised= True,download=True
                                               )

food_classes = food_info.features["label"].num_classes


IMG_SIZE = 224

input_size = (IMG_SIZE,IMG_SIZE)

dogs_train = dogs_train.map(lambda image, label: (tf.image.resize(image, input_size), label))
dogs_test = dogs_test.map(lambda image, label: (tf.image.resize(image, input_size), label))

food_train = food_train.map(lambda image, label: (tf.image.resize(image, input_size), label))
food_test = food_test.map(lambda image, label: (tf.image.resize(image, input_size), label))



def format_label (label):
    label_string = label_info.int2str(label)
    return label_string.split("-")[1]

label_info = dogs_info.features["label"]
for i, (image, label) in enumerate(dogs_train.take(9)):
    ax = plt.subplot(3, 3, i + 1)
    plt.imshow(image.numpy().astype("uint8"))
    plt.title("{}".format(format_label(label)))
    plt.axis("off")

plt.show()

label_info = food_info.features["label"]
for i, (image, label) in enumerate(food_train.take(9)):
    ax = plt.subplot(3, 3, i + 1)
    plt.imshow(image.numpy().astype("uint8"))
    plt.title("{}".format(label_info.int2str(label)))
    plt.axis("off")

plt.show()



from tensorflow.keras.layers.experimental import preprocessing
from tensorflow.keras.models import Sequential
from tensorflow.keras import layers

img_aug = Sequential(
    [
        preprocessing.RandomRotation(factor=0.2),
        preprocessing.RandomTranslation(height_factor=0.125,width_factor=0.125),
        preprocessing.RandomFlip(),
        preprocessing.RandomContrast(factor=0.125)
    ],
    name="img_augmentation"
)

label_info = dogs_info.features["label"]
for image, label in dogs_train.take(1):
    for i in range(9):
        ax = plt.subplot(3, 3, i + 1)
        augmented_img = img_aug(tf.expand_dims(image, axis=0)) #expand input to 4D tensor, so it can be preprocessed by Keras
        plt.imshow(augmented_img[0].numpy().astype("uint8"))
        plt.title("{}".format(format_label(label)))
        plt.axis("off")

plt.show()

label_info = food_info.features["label"]
for image, label in food_train.take(1):
    for i in range(9):
        ax = plt.subplot(3, 3, i + 1)
        augmented_img = img_aug(tf.expand_dims(image, axis=0))
        plt.imshow(augmented_img[0].numpy().astype("uint8"))
        plt.title("{}".format(label_info.int2str(label)))
        plt.axis("off")

plt.show()


def input_category (image,label):
    label = tf.one_hot(label,num_of_classes)
    return image,label


batch_size = 64

num_of_classes = dogs_classes
dogs_train = dogs_train.map(input_category)
dogs_train = dogs_train.batch(batch_size= batch_size, drop_remainder = True)

dogs_test = dogs_test.map(input_category)
dogs_test = dogs_test.batch(batch_size= batch_size, drop_remainder = True)

num_of_classes = food_classes
food_train = food_train.map(input_category)
food_train = food_train.batch(batch_size= batch_size, drop_remainder = True)

food_test = food_test.map(input_category)
food_test = food_test.batch(batch_size= batch_size, drop_remainder = True)


dogs_info = dogs_info.features["label"]
food_info = food_info.features["label"]

from sam import sam_train_step



def model_build_no_sam(classes_num):
    inputs = layers.Input(shape=(IMG_SIZE,IMG_SIZE,3))
    x = img_aug(inputs)
    model = EfficientNetB0(include_top=False, input_tensor=x,weights="imagenet")

    model.trainable = False

    x = layers.GlobalAveragePooling2D(name="avg_pool")(model.output)
    x = layers.BatchNormalization()(x)

    dropout = 0.5
    x = layers.Dropout(dropout, name="top_dropout")(x)
    outputs_layer = layers.Dense(classes_num, activation="softmax", name="prediction")(x)

    model = tf.keras.Model(inputs,outputs_layer, name="EfficientNetB0")

    for layer in model.layers[-20:]:
        if not isinstance(layer,layers.BatchNormalization):
            layer.trainable = True

    #to_test: Nadam,SGDM,RMSP
    optimizer = tf.keras.optimizers.Adam(learning_rate=1e-3)

    model.compile(optimizer = optimizer, loss="categorical_crossentropy", metrics=["accuracy"])


    return model

class YourModel(tf.keras.Model):
    def train_step(self, data):
        return sam_train_step(self, data)

def model_build_sam(classes_num):
    inputs = layers.Input(shape=(IMG_SIZE,IMG_SIZE,3))
    x = img_aug(inputs)
    model = EfficientNetB0(include_top=False, input_tensor=x,weights="imagenet")

    model.trainable = False

    x = layers.GlobalAveragePooling2D(name="avg_pool")(model.output)
    x = layers.BatchNormalization()(x)

    dropout = 0.5
    x = layers.Dropout(dropout, name="top_dropout")(x)
    outputs_layer = layers.Dense(classes_num, activation="softmax", name="prediction")(x)

    model = tf.keras.Model(inputs,outputs_layer, name="EfficientNetB0")

    for layer in model.layers[-20:]:
        if not isinstance(layer,layers.BatchNormalization):
            layer.trainable = True

    model = YourModel(inputs,outputs_layer)
    optimizer = tf.keras.optimizers.Adam(learning_rate=1e-4)

    model.compile(optimizer = optimizer, loss="categorical_crossentropy", metrics=["accuracy"])


    return model

strategy = tf.distribute.MirroredStrategy()

with strategy.scope():
    dogs_model_no_sam = model_build_no_sam(classes_num=dogs_classes)
    food_model_no_sam = model_build_no_sam(classes_num=food_classes)
    dogs_model_sam = model_build_sam(classes_num=dogs_classes)
    food_model_sam = model_build_sam(classes_num=food_classes)
epoch = 50

print(dogs_model_no_sam.summary())
print("===================================================================")
print(dogs_model_sam.summary())

def plot_hist(hist):
    plt.plot(hist.history["accuracy"])
    plt.plot(hist.history["val_accuracy"])
    plt.title("model accuracy")
    plt.ylabel("accuracy")
    plt.xlabel("epoch")
    plt.legend(["train", "validation"], loc="upper left")


def plot_hist_loss(hist):
    plt.plot(hist.history["loss"])
    plt.plot(hist.history["val_loss"])
    plt.title("model loss")
    plt.ylabel("loss")
    plt.xlabel("epoch")
    plt.legend(["train", "validation"], loc="upper left")

model_number = 1
stopper = tf.keras.callbacks.EarlyStopping(monitor='val_loss', mode='min',patience=5)
logger = tf.keras.callbacks.CSVLogger(
    'prediction_model_'+str(model_number)+'.csv', separator=',', append=False
)
#Stanford Dogs model
prediction_dog_no_sam = dogs_model_no_sam.fit(dogs_train, epochs=epoch, validation_data=dogs_test,callbacks=[stopper,logger])

model_number = 2
prediction_dog_sam = dogs_model_sam.fit(dogs_train, epochs=epoch, validation_data=dogs_test, callbacks=[stopper,logger])

plot_hist(prediction_dog_no_sam)
plt.savefig('dogs_prediction_without_SAM')
plt.show()
plot_hist_loss(prediction_dog_no_sam)
plt.savefig('dogs_prediction_without_SAM_loss')
plt.show()

plot_hist(prediction_dog_sam)
plt.savefig('dogs_prediction_with_SAM')
plt.show()
plot_hist_loss(prediction_dog_sam)
plt.savefig('dogs_prediction_with_SAM_loss')
plt.show()
# tf.keras.utils.plot_model(prediction_dog_sam,to_file="EffnetModel.png")

#Food101 model
model_number=3
prediction_food_no_sam = food_model_no_sam.fit(food_train, epochs=epoch, validation_data=food_test,callbacks=[stopper,logger])
model_number=4
prediction_food_sam = food_model_sam.fit(food_train, epochs=epoch, validation_data=food_test,callbacks=[stopper,logger])


plot_hist(prediction_food_no_sam)
plt.savefig('food101_prediction_without_SAM')
plt.show()
plot_hist_loss(prediction_food_no_sam)
plt.savefig('food101_prediction_without_SAM_loss')
plt.show()

plot_hist(prediction_food_sam)
plt.savefig('food101_prediction_with_SAM')
plt.show()
plot_hist_loss(prediction_food_sam)
plt.savefig('food101_prediction_with_SAM_loss')
plt.show()
