import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D
from tensorflow.keras.layers import Activation, Dropout, Flatten, Dense
from tensorflow.keras import backend as K
import numpy as np
import pandas as pd

#use gpu
with tf.device('/gpu:0'):

    img_width = 150
    img_height = 150

    if K.image_data_format() == 'channels_first':
        input_shape = (3, img_width, img_height)
    else:
        input_shape = (img_width, img_height, 3)


    #define model
    model = Sequential()
    model.add(Conv2D(32, (3, 3), input_shape=input_shape))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Conv2D(32, (3, 3)))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Conv2D(64, (3, 3)))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Flatten())
    model.add(Dense(64))
    model.add(Activation('relu'))
    model.add(Dropout(0.5))
    model.add(Dense(35))
    model.add(Activation('softmax'))

    #dense output should be number of classes


    print(model.summary())

    #categorical_crossentropy for multi class
    #try adam optimizer, prev 'rmsprop'

    model.compile(loss='categorical_crossentropy',
        optimizer='adam',
        metrics=['accuracy']
    )  


    train_datagen = ImageDataGenerator(
        rescale=1./255,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True
    )

    test_datagen = ImageDataGenerator(rescale=1./255)

    valid_datagen = ImageDataGenerator(
        rescale=1./255,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True
    )


    print("Train: ")

    train_generator = train_datagen.flow_from_directory(
        directory=r"C:\Users\Sakshi\Documents\Projects\FinalYear\Code\Split Dataset\train",
        target_size=(150, 150),
        color_mode="rgb",
        batch_size=32,
        class_mode="categorical",
        shuffle=True,
        seed=42
        )

    print("Validation: ")

    valid_generator = valid_datagen.flow_from_directory(
        directory=r"C:\Users\Sakshi\Documents\Projects\FinalYear\Code\Split Dataset\val",
        target_size=(150, 150),
        color_mode="rgb",
        batch_size=32,
        class_mode="categorical",
        shuffle=True,
        seed=42
    )
    print("Test: ")

    test_generator = test_datagen.flow_from_directory(
        directory=r"C:\Users\Sakshi\Documents\Projects\FinalYear\Code\Split Dataset\test",
        target_size=(150, 150),
        color_mode="rgb",
        batch_size=1,
        class_mode=None,
        shuffle=False,
        seed=42
    )

    STEP_SIZE_TRAIN=train_generator.n//train_generator.batch_size
    STEP_SIZE_VALID=valid_generator.n//valid_generator.batch_size

    print("Step size:")
    print("--- Train: " + str(STEP_SIZE_TRAIN))
    print("---validation: " + str(STEP_SIZE_VALID))
    
    #np.asarray(valid_generator)    
    model.fit_generator(train_generator,
                    steps_per_epoch=STEP_SIZE_TRAIN,
                    validation_data=valid_generator,
                    validation_steps=STEP_SIZE_VALID,
                    epochs=10
    )

    model.evaluate_generator(generator=valid_generator)

    test_generator.reset()


    print("\n\n Prediction: ")
    pred = model.predict_generator(test_generator, verbose=1)

    predicted_class_indices = np.argmax(pred, axis=1)

    labels = (train_generator.class_indices)
    np.save(r"C:\Users\Sakshi\Documents\Projects\FinalYear\Code\classes", labels)
    labels = dict((v, k) for k, v in labels.items())
    predictions = [labels[k] for k in predicted_class_indices]

    print("\n\nGenerating CSV 'results.csv'")
    filenames=test_generator.filenames
    results=pd.DataFrame({"Filename":filenames,
                      "Predictions":predictions})
    results.to_csv("results.csv",index=False)

    model.save(r"C:\Users\Sakshi\Documents\Projects\FinalYear\Code\best_model.h5")

print("\n\nExiting program\n\n")


