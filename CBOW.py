import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Embedding
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

# a. Data preparation
corpus = [
    "the quick brown fox jumps over the lazy dog",
    "all that glitters is not gold",
    "the earth revolves around the sun"
]

# Tokenize the corpus
tokenizer = Tokenizer()
tokenizer.fit_on_texts(corpus)
vocab_size = len(tokenizer.word_index) + 1

# b. Generate training data
window_size = 2
X = []
y = []
for text in corpus:
    tokens = tokenizer.texts_to_sequences([text])[0]
    for i in range(len(tokens)):
        context = tokens[max(0, i - window_size):i] + tokens[i+1:i + window_size + 1]
        target = tokens[i]
        X.append(context)
        y.append(target)

X = pad_sequences(X, maxlen=2 * window_size)
y = np.array(y)

# c. Train model
model = Sequential()
model.add(Embedding(vocab_size, 100, input_length=2 * window_size))
model.add(tf.keras.layers.Lambda(lambda x: tf.reduce_mean(x, axis=1)))
model.add(Dense(vocab_size, activation='softmax'))
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

model.fit(X, y, epochs=50, batch_size=32, verbose=2)

# d. Output
def predict_word(context_words):
    context_idx = [tokenizer.word_index[w] for w in context_words]
    context_input = pad_sequences([context_idx], maxlen=2 * window_size)
    predicted_index = np.argmax(model.predict(context_input)[0])
    predicted_word = None
    for word, index in tokenizer.word_index.items():
        if index == predicted_index:
            predicted_word = word
            break
    return predicted_word

context = ["the", "quick"]
predicted_word = predict_word(context)
print(f"Given the context '{' '.join(context)}', the predicted word is: {predicted_word}")