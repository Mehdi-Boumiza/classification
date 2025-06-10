import tensorflow as tf
from tensorflow.keras.datasets import cifar10
(x_train,y_train) , (x_test,y_test) = cifar10.load_data()
x_train = x_train.astype('float32')/225.0
x_test = x_test.astype('float32')/225.0

y_train = tf.keras.utils.to_categorical(y_train,10)
y_test = tf.keras.utils.to_categorical(y_test,10)

# print(x_train.shape, y_train.shape) 
# print(x_test.shape, y_test.shape)    

from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input,Conv2D, MaxPooling2D, Flatten, Dense ,Dropout, BatchNormalization

input_layer = Input(shape =(32,32,3))

x = Conv2D(64,(3,3),activation = 'relu',padding = 'same')(input_layer)
x = Conv2D(64,(3,3),activation = 'relu',padding = 'same')(x)
x = BatchNormalization()(x)
x = MaxPooling2D((2,2))(x)
x = Dropout(0.25)(x)

x = Conv2D(128,(3,3),activation = 'relu',padding ='same')(x)
x = Conv2D(128,(3,3), activation = 'relu',padding = 'same')(x)
x = BatchNormalization()(x)
x = MaxPooling2D((2,2))(x)
x = Dropout(0.25)(x)

x = Conv2D(256,(3,3),activation = 'relu',padding = 'same')(x)
x = Conv2D(256,(3,3),activation= 'relu',padding = 'same')(x)
x = BatchNormalization()(x)
x = MaxPooling2D((2,2), strides = (2,2))(x)
x = Dropout(0.25)(x)

x = Conv2D(512,(3,3), activation = 'relu', padding = 'same')(x)
x = Conv2D(512,(3,3), activation = 'relu', padding = 'same')(x)
x = Conv2D(512,(3,3), activation = 'relu', padding = 'same')(x)
x = BatchNormalization()(x)
x = MaxPooling2D((2,2))(x)
x = Dropout(0.25)(x)

x = Conv2D(512,(3,3),activation = 'relu', padding = 'same')(x)
x = Conv2D(512,(3,3), activation = 'relu', padding = 'same')(x)
x = Conv2D(512,(3,3),activation = 'relu', padding = 'same')(x)
x = BatchNormalization()(x)
x = MaxPooling2D((2,2))(x)
x = Dropout(0.25)(x)

x = Flatten()(x)
x = Dense(4096,activation = 'relu')(x)
x = Dense(4096,activation = 'relu')(x)
x = Dropout(0.5)(x)
output_layer = Dense(10,activation = 'softmax')(x)

model = Model(inputs = input_layer, outputs = output_layer)

model.compile(
    optimizer = 'adam',
    loss = 'categorical_crossentropy',
    metrics = ['accuracy']
)

from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
stop = EarlyStopping(
    monitor = 'val_loss',
    patience = 5,
    restore_best_weights = True,
)
lr = ReduceLROnPlateau(
    monitor = 'val_loss',
    patience = 3,
    verbose = 1,
    factor = 0.5,
)


model.fit(x_train,y_train,epochs = 50,validation_split = 0.1,batch_size = 128, callbacks = [stop,lr])
test_loss, test_acc = model.evaluate(x_test,y_test)
print(f"test accuracy: {test_acc:.3f}")


model.save("my_cifar10_model.h5")  

import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import matplotlib.pyplot as plt

class_names = ['airplane','automobile','bird','cat','deer','dog','frog','horse','ship','truck']
model = load_model("my_cifar10_model.h5")

img_path = "/Users/mehdiboumiza/Documents/functional API/download (15).jpeg"
img = image.load_img(img_path,target_size = (32,32))
img_array = image.img_to_array(img).astype('float32') / 225.0
img_array = np.expand_dims(img_array,axis =0)

prediction = model.predict(img_array)
guess = np.argmax(prediction[0])
confidence = np.max(prediction[0])

plt.imshow(img)
plt.title(f"predicted class: {class_names[guess]} ({confidence:.2f})")
plt.axis('off')
plt.show()

