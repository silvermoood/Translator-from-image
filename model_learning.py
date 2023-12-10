import idx2numpy
import numpy as np
import keras
from keras.models import Sequential # позволяет создавать модели нейронных сетей последовательно
from keras.layers import Convolution2D, MaxPooling2D, Dropout, Flatten, Dense # сверточные слои (Convolution2D), слои пулинга (MaxPooling2D), слои регуляризации (Dropout), слой выравнивания (Flatten), полносвязные слои (Dense), слои изменения формы (Reshape), слой долгой краткосрочной памяти (LSTM) и нормализации по пакету (BatchNormalization).

# 62 различных символа в датасете EMNIST
emnist_labels = [48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 65, 66, 67, 68, 69, 70, 71, 72, 73, 74, 75, 76, 77, 78, 79, 80, 81, 82, 83, 84, 85, 86, 87, 88, 89, 90, 97, 98, 99, 100, 101, 102, 103, 104, 105, 106, 107, 108, 109, 110, 111, 112, 113, 114, 115, 116, 117, 118, 119, 120, 121, 122]

# Нейронная сеть на вход получает изображения 28х28, и имеет 62 выхода. Послераспознавания «1» будет на соответствующем выходе сети
# Выделяет определенные признаки изображения (количество фильтров 32 и 64), к «выходу» которой подсоединена «линейная» сеть MLP, формирующая окончательный результат

def emnist_model():
    model = Sequential() # объект модели с последовательными слоями
    model.add(Convolution2D(filters=32, kernel_size=(3, 3), padding='valid', input_shape=(28, 28, 1), activation='relu')) # сверточный слой с 32 фильтрами, ядром размера (3, 3), функцией активации ReLU и указанием размера входных данных
    model.add(Convolution2D(filters=64, kernel_size=(3, 3), activation='relu')) # еще один сверточной слоя с 64 фильтрами и аналогичными параметрами
    model.add(MaxPooling2D(pool_size=(2, 2))) # слой пулинга с размером пула (2, 2)
    model.add(Dropout(0.25)) # слой Dropout для регуляризации с вероятностью отсева 25%
    model.add(Flatten()) # выравнивание данных перед подачей их на полносвязные слои
    model.add(Dense(512, activation='relu')) # полносвязный слой с 512 нейронами и функцией активации ReLU
    model.add(Dropout(0.5)) #  слой Dropout для регуляризации с вероятностью отсева 50%
    model.add(Dense(len(emnist_labels), activation='softmax')) # полносвязный слой с количеством нейронов, соответствующим количеству emnist_labels
    model.compile(loss='categorical_crossentropy', optimizer='adadelta', metrics=['accuracy']) # и компиляция модели с использованием категориальной кросс-энтропии в качестве функции потерь, оптимизатора Adadelta и метрики точности
    return model

# подключаем датасет EMNIST и выбираем данные для обучения и для тестирования
emnist_path = "C:\\Users\\sereb\\PycharmProjects\\pythonProject\\Neural_Networks\\gzip"
X_train = idx2numpy.convert_from_file(emnist_path + '\\emnist-byclass-train-images-idx3-ubyte')
y_train = idx2numpy.convert_from_file(emnist_path + '\\emnist-byclass-train-labels-idx1-ubyte')
X_test = idx2numpy.convert_from_file(emnist_path + '\\emnist-byclass-test-images-idx3-ubyte')
y_test = idx2numpy.convert_from_file(emnist_path + '\\emnist-byclass-test-labels-idx1-ubyte')
# изменение размера на 28*28 и глубиной 1 - чб
X_train = np.reshape(X_train, (X_train.shape[0], 28, 28, 1))
X_test = np.reshape(X_test, (X_test.shape[0], 28, 28, 1))

# нормализация данных
X_train = X_train.astype(np.float32)
X_train /= 255.0
X_test = X_test.astype(np.float32)
X_test /= 255.0
x_train_cat = keras.utils.to_categorical(y_train, len(emnist_labels))
y_test_cat = keras.utils.to_categorical(y_test, len(emnist_labels))


learning_rate_reduction = keras.callbacks.ReduceLROnPlateau(monitor='val_accuracy', patience=3, verbose=1, factor=0.5, min_lr=0.00001)
model = emnist_model()
model.fit(X_train, x_train_cat, validation_data=(X_test, y_test_cat), callbacks=[learning_rate_reduction], batch_size=64, epochs=30)
model.save('emnist_letters.h5')
