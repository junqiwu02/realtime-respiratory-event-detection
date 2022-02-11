# %%
from preprocess import *
from tensorflow import keras
from tensorflow.keras.layers import InputLayer, Dense
import analysis
from sklearn.utils import shuffle

# %%
data = Preprocess()
X_test, y_test = data.get_test()
pickle_dump(X_test, 'data/x_test_mfcc.pkl')
pickle_dump(y_test, 'data/y_test_mfcc.pkl')

X_train, y_train = data.get_train()
pickle_dump(X_train, 'data/x_train_mfcc.pkl')
pickle_dump(y_train, 'data/y_train_mfcc.pkl')

X_val, y_val = data.get_val()
pickle_dump(X_val, 'data/x_val_mfcc.pkl')
pickle_dump(y_val, 'data/y_val_mfcc.pkl')

# %%
X_test = pickle_load('data/x_test_mfcc.pkl')
y_test = pickle_load('data/y_test_mfcc.pkl')

X_train = pickle_load('data/x_train_mfcc.pkl')
y_train = pickle_load('data/y_train_mfcc.pkl')

X_val = pickle_load('data/x_val_mfcc.pkl')
y_val = pickle_load('data/y_val_mfcc.pkl')

# %%
X_train, y_train = resample(X_train, y_train)
X_train, y_train = shuffle(X_train, y_train)

# %%
model = keras.Sequential()
model.add(InputLayer(input_shape=(FEAT_DIM,)))
model.add(Dense(128, activation='tanh'))
model.add(Dense(256, activation='tanh'))
model.add(Dense(128, activation='tanh'))
model.add(Dense(1, activation='sigmoid'))
print(model.summary())

# %%
model.compile(
    loss=keras.losses.BinaryCrossentropy(from_logits=False),
    optimizer=keras.optimizers.Adam(learning_rate=0.001),
    metrics=["accuracy"],
)

# %%
model.fit(X_train, y_train, batch_size=64, epochs=100, validation_data=(X_val, y_val))


# %%
analysis.show_stats(y_test, model.predict(X_test))

# %%
