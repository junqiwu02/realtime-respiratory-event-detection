# %%
from preprocess import *
from tensorflow import keras
from tensorflow.keras.layers import InputLayer, LSTM, Dense, GRU
import analysis
from sklearn.utils import shuffle

# %%
data = Preprocess()
# X_train, y_train = data.get_train()
# pickle_dump(X_train, 'data/x_train_mfcc.pkl')
# pickle_dump(y_train, 'data/y_train_mfcc.pkl')

X_val, y_val = data.get_val()
pickle_dump(X_val, 'data/x_val_mfcc.pkl')
pickle_dump(y_val, 'data/y_val_mfcc.pkl')

# %%
X_train = pickle_load('data/x_train_mfcc.pkl')
y_train = pickle_load('data/y_train_mfcc.pkl')

X_val = pickle_load('data/x_val_mfcc.pkl')
y_val = pickle_load('data/y_val_mfcc.pkl')

# %%
# X_train, y_train = resample(X_train, y_train)
X_train, y_train = shuffle(X_train, y_train)

# y_train = one_hot(y_train)
# y_val = one_hot(y_val)

# %%
model = keras.Sequential()
model.add(InputLayer(input_shape=(40,)))
# model.add(GRU(256, return_sequences=True, activation='tanh'))
# model.add(GRU(256, return_sequences=True, activation='tanh'))
# model.add(GRU(256, activation='tanh'))
# model.add(Dense(1, activation='sigmoid'))
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
# model.fit(X_train, y_train, sample_weight=sample_weight, batch_size=50, epochs=100)
model.fit(X_train, y_train, batch_size=64, epochs=500, validation_data=(X_val, y_val))


# %%
analysis.show_stats(y_val, model.predict(X_val))

# %%
