from twitter_preproc import *
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.models import Sequential
from keras.layers import Activation, Dense, Dropout, Embedding, LSTM
from keras import utils
from keras.callbacks import ReduceLROnPlateau, EarlyStopping, TensorBoard
import gensim
import time
from sklearn.model_selection import train_test_split
import tensorflow
import pandas as pd


# WORD2VEC
W2V_SIZE = 300
W2V_WINDOW = 7
W2V_EPOCH = 32
W2V_MIN_COUNT = 3

# KERAS
SEQUENCE_LENGTH = 300
EPOCHS = 8
BATCH_SIZE = 512
DATASET_COLUMNS = ["target", "text"]

dataset_path = 'datasets/twitter_train_big.csv'

DATASET_ENCODING = "ISO-8859-1"
TRAIN_SIZE = 0.7

df = pd.read_csv(dataset_path, encoding=DATASET_ENCODING, names=DATASET_COLUMNS)

def labelTransform(l):
    if l == 4:
        return 1
    else:
        return 0
df = df[['target', 'text']]


print(len(df))

print(df.head(5))

df.target = df.target.apply(lambda x: labelTransform(x))
df.text = df.text.apply(lambda x: preprocess(x))

df_train, df_test = train_test_split(df, test_size=1-TRAIN_SIZE, random_state=42)

print("TRAIN size:", len(df_train))
print("TEST size:", len(df_test))

# df_train.to_csv('datasets/twitter_train_big.csv', index=False, encoding=DATASET_ENCODING ,header=False)


documents = [_text.split() for _text in df_train.text]

w2v_model = gensim.models.word2vec.Word2Vec(size=W2V_SIZE,
                                            window=W2V_WINDOW,
                                            min_count=W2V_MIN_COUNT,
                                            workers=8)
w2v_model.build_vocab(documents)

tokenizer = Tokenizer()
tokenizer.fit_on_texts(df_train.text)

vocab_size = len(tokenizer.word_index) + 1
print("Total words", vocab_size)

x_train = pad_sequences(tokenizer.texts_to_sequences(df_train.text), maxlen=SEQUENCE_LENGTH)
x_test = pad_sequences(tokenizer.texts_to_sequences(df_test.text), maxlen=SEQUENCE_LENGTH)
y_train = np.array(df_train.target.tolist())
y_test = np.array(df_test.target.tolist())
y_train = y_train.reshape(-1,1)
y_test = y_test.reshape(-1,1)

print("x_train", x_train.shape)
print("y_train", y_train.shape)
print()
print("x_test", x_test.shape)
print("y_test", y_test.shape)

embedding_matrix = np.zeros((vocab_size, W2V_SIZE))
for word, i in tokenizer.word_index.items():
  if word in w2v_model.wv:
    embedding_matrix[i] = w2v_model.wv[word]
print(embedding_matrix.shape)
embedding_layer = Embedding(vocab_size, W2V_SIZE,
                            weights=[embedding_matrix],
                            input_length=SEQUENCE_LENGTH,
                            trainable=False)

model = Sequential()
model.add(embedding_layer)
model.add(Dropout(0.5))
model.add(LSTM(64, dropout=0.2, recurrent_dropout=0.2))
model.add(Dense(1, activation='sigmoid'))

model.compile(loss='binary_crossentropy',
              optimizer="adam",
              metrics=['accuracy'])
model.summary()


tensorboard = TensorBoard(log_dir='./logs/lstm', histogram_freq=0,
                          write_graph=True, write_images=False)
callbacks = [ ReduceLROnPlateau(monitor='val_loss', patience=5, cooldown=0),
              EarlyStopping(monitor='val_acc', min_delta=1e-4, patience=5),
              tensorboard]

history = model.fit(x_train, y_train,
                    batch_size=BATCH_SIZE,
                    epochs=EPOCHS,
                    validation_split=0.,
                    verbose=1,
                    callbacks=callbacks)

score = model.evaluate(x_test, y_test, batch_size=BATCH_SIZE)
print()
print("ACCURACY:", score[1])
print("LOSS:", score[0])


def predict(text, include_neutral=True):
    start_at = time.time()
    # Tokenize text
    x_test = pad_sequences(tokenizer.texts_to_sequences([text]), maxlen=SEQUENCE_LENGTH)
    # Predict
    score = model.predict([x_test])[0]
    # Decode sentiment
    label = 0 if score < 0.5 else 1

    return {"label": label, "score": float(score),
       "elapsed_time": time.time()-start_at}


print(predict("I love the music"))
print(predict("I hate the rain"))

model.save('twi_lstm.h5')
w2v_model.save('w2v.w2v')
