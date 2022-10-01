from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import MaxPooling2D
from tensorflow.keras.layers import Conv2D
from tensorflow.keras.layers import BatchNormalization
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Flatten
from tensorflow.keras.preprocessing.image import ImageDataGenerator, DirectoryIterator
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
import matplotlib.pyplot as plt
import os.path as path

model = Sequential()
model.add(Conv2D(64,(3,3), activation = "relu", padding="same", input_shape = (224,224,3)))
model.add(BatchNormalization())
model.add(Conv2D(64,(3,3), padding="same", activation="relu"))
model.add(BatchNormalization())
model.add(MaxPooling2D())
model.add(Conv2D(128,(3,3), padding="same", activation="relu"))
model.add(BatchNormalization())
model.add(Conv2D(128,(3,3), padding="same", activation="relu"))
model.add(BatchNormalization())
model.add(MaxPooling2D())
model.add(Conv2D(256,(3,3), padding="same", activation="relu"))
model.add(BatchNormalization())
model.add(Conv2D(256,(3,3), padding="same", activation="relu"))
model.add(BatchNormalization())
model.add(Conv2D(256,(3,3), padding="same", activation="relu"))
model.add(BatchNormalization())
model.add(MaxPooling2D())
model.add(Conv2D(512,(3,3), padding="same", activation="relu"))
model.add(BatchNormalization())
model.add(Conv2D(512,(3,3), padding="same", activation="relu"))
model.add(BatchNormalization())
model.add(Conv2D(512,(3,3), padding="same", activation="relu"))
model.add(BatchNormalization())
model.add(MaxPooling2D())
model.add(Conv2D(512,(3,3), padding="same", activation="relu"))
model.add(BatchNormalization())
model.add(Conv2D(512,(3,3), padding="same", activation="relu"))
model.add(BatchNormalization())
model.add(Conv2D(512,(3,3), padding="same", activation="relu"))
model.add(BatchNormalization())
model.add(MaxPooling2D())
model.add(Flatten())
model.add(Dense(4096,activation= "relu"))
model.add(Dense(4096,activation= "relu"))
model.add(Dense(6,activation='softmax'))
model.summary()
model.compile(optimizer="adam",loss='categorical_crossentropy',metrics=['accuracy'])
train_datagen = ImageDataGenerator( rescale = 1.0/255.0,
                                    rotation_range=40,
                                    width_shift_range=0.2,
                                    height_shift_range=0.2,
                                    shear_range=0.2,
                                    zoom_range=0.2,
                                    horizontal_flip=True,
                                    fill_mode='nearest')
test_datagen  = ImageDataGenerator( rescale = 1.0/255)
train_path = path.join(path.abspath(path.dirname(__file__)), "./trainFace")
train_generator = DirectoryIterator(train_path,
                                    train_datagen,
                                    batch_size=1,
                                    class_mode='categorical',
                                    target_size=(224, 224))    
val_path = path.join(path.abspath(path.dirname(__file__)), "./validationFace") 
validation_generator = DirectoryIterator( val_path,
                                          test_datagen,
                                          batch_size=1,
                                          class_mode  = 'categorical',
                                          target_size = (224, 224))
test_path = path.join(path.abspath(path.dirname(__file__)), "./testFace")
test_generator = DirectoryIterator( test_path,
                                    test_datagen,
                                    batch_size=1,
                                    class_mode  = 'categorical',
                                    target_size = (224, 224))
print(train_generator.class_indices.keys())
label = train_generator.class_indices.keys()
f = open('./public/label.txt', mode='w', encoding='utf-8')
for i  in label:
      f.write(i+"\n")
      print(i)
es = EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience= 9)
mc = ModelCheckpoint(path.join(path.abspath(path.dirname(__file__))+ "\model_face.h5"), monitor='val_accuracy', mode='max', verbose=1, save_best_only=True)

model_saved=model.fit(
      train_generator,
      epochs=5000,
      validation_data=validation_generator,
      callbacks=[mc, es]
)

print(model.evaluate(test_generator))
acc = model_saved.history['accuracy']
val_acc = model_saved.history['val_accuracy']
loss = model_saved.history['loss']
val_loss = model_saved.history['val_loss']

epochs = range(len(acc))
plt.figure(figsize=(10, 6))

plt.plot(epochs, acc, 'r', label='Training Accuracy')
plt.plot(epochs, val_acc, 'b', label='Validation Accuracy')
plt.plot(epochs, loss, 'g', label='Training Loss')
plt.plot(epochs, val_loss, 'y', label='Validation Loss')

plt.title('Traing and Validation, Accuracy and Loss')
plt.legend(loc=0)
plt.show()

import tensorflowjs as tf
from keras.models import Sequential,load_model
model = load_model(path.join(path.abspath(path.dirname(__file__)), "./model_face.h5"))
tf.converters.save_keras_model(model, path.join(path.abspath(path.dirname(__file__)), "./modelface"))
