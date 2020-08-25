import tensorflow as t
import tensorflow.keras as k
import matplotlib.pyplot as p

# x input, y label
minst = k.datasets.mnist
(x_train,y_train), (x_test, y_test) = minst.load_data()
# p.imshow(x_train[0], cmap=p.cm.binary)
x_train = k.utils.normalize(x_train, axis=1)
x_test = k.utils.normalize(x_test, axis=1)
# p.imshow(x_train[0], cmap=p.cm.binary)
model = k.models.Sequential()
model.add(k.layers.Flatten())
model.add(k.layers.Dense(1024 ,activation=t.nn.relu))
model.add(k.layers.Dense(128 ,activation=t.nn.relu))
model.add(k.layers.Dense(10,activation=t.nn.softmax))
model.compile(optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"])
model.fit(x_train,y_train, epochs=3)
val_loss, val_acc = model.evaluate(x_test,y_test)

model.save('number_ai.dnn.data')
model.predict([x_test])
