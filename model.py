from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelBinarizer
from sklearn.svm import SVC
from afinn import Afinn
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.feature_extraction.text import TfidfVectorizer
from transformers import pipeline
import json
from transformers import AutoTokenizer, TFAutoModelForSequenceClassification
import numpy as np
import tensorflow as tf


def load_data(train_path, test_path):
    with open(f"train.json", "r") as f:
        train = json.load(f)
    with open(f"test.json", "r") as f:
        test = json.load(f)

    X_train = [sample["review"] for sample in train]
    X_test = [sample["review"] for sample in test]
    y_train = [sample["label"] for sample in train]
    y_test = [sample["label"] for sample in test]

    return X_train, y_train, X_test, y_test


def identity(x):
    return x


def print_scores(scores, score_name, labels):
    """Print the metrics along with their class"""
    print(f"{score_name} by class")
    for score, label in zip(scores, labels):
        print(f"{label}: {score:.3f}")
    print("\n")


def get_scores(y_test, Y_pred, model_name):
    accuracy = accuracy_score(y_test, Y_pred)
    f_score = f1_score(y_test, Y_pred, average=None, labels=["pos", "neg"])
    precision = precision_score(y_test, Y_pred, average=None, labels=["pos", "neg"])
    recall = recall_score(y_test, Y_pred, average=None, labels=["pos", "neg"])

    print("Model: ", model_name)
    print(f"Accuracy: {accuracy:.3f}")
    print_scores(f_score, "F Score", ["pos", "neg"])
    print_scores(precision, "Precision", ["pos", "neg"])
    print_scores(recall, "Recall", ["pos", "neg"])


def get_callbacks(epochs):
    """Create the early stopping and learning rate scheduler callbacks"""
    early_stopping = tf.keras.callbacks.EarlyStopping(
        monitor="val_loss", patience=int(epochs // 5)
    )
    reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(
        monitor="val_loss",
        factor=0.1,
        patience=int(epochs // 8),
        verbose=0,
        mode="auto",
        min_delta=0.001,
        cooldown=0,
        min_lr=0.00000001,
    )

    return [early_stopping, reduce_lr]


def read_embeddings(embeddings_file):
    """Read in word embeddings from file and save as numpy array"""
    embeddings = open(embeddings_file, "r", encoding="utf-8").read()
    return {
        line.split()[0]: np.array(line.split()[1:], dtype=float)
        for line in embeddings.split("\n")[:-1]
    }


def get_emb_matrix(voc, emb):
    """Get embedding matrix given vocab and the embeddings"""
    num_tokens = len(voc) + 2
    word_index = dict(zip(voc, range(len(voc))))
    # Bit hacky, get embedding dimension from the word "the"
    embedding_dim = len(emb["the"])
    # Prepare embedding matrix to the correct size
    embedding_matrix = np.zeros((num_tokens, embedding_dim))
    for word, i in word_index.items():
        embedding_vector = emb.get(word)
        if embedding_vector is not None:
            # Words not found in embedding index will be all-zeros.
            embedding_matrix[i] = embedding_vector
    # Final matrix with pretrained embeddings that we can feed to embedding layer
    return embedding_matrix


def test_afinn_model(X, y):
    afinn = Afinn()
    afinn_scores = [afinn.score(review) for review in X]
    afinn_preds = ["neg" if score < 0 else "pos" for score in afinn_scores]

    get_scores(y, afinn_preds, "AFINN")


def train_test_ML_model(X_train, y_train, X_test, y_test, model, model_name):
    vec = TfidfVectorizer(tokenizer=identity, preprocessor=identity)

    classifier = Pipeline([("vec", vec), ("cls", model)])
    classifier.fit(X_train, y_train)

    Y_pred = classifier.predict(X_test)
    get_scores(y_test, Y_pred, model_name)


def train_test_naive_bayes(X_train, y_train, X_test, y_test):
    model = MultinomialNB()
    train_test_ML_model(X_train, y_train, X_test, y_test, model, "Naive Bayes")


def train_test_logistic_reg(X_train, y_train, X_test, y_test):
    model = LogisticRegression()
    train_test_ML_model(X_train, y_train, X_test, y_test, model, "Logistic Regression")


def train_test_SVM(X_train, y_train, X_test, y_test):
    model = SVC()
    train_test_ML_model(X_train, y_train, X_test, y_test, model, "SVM")


def test_base_distilbert(X_test, y_test):
    classifier = pipeline("sentiment-analysis", max_length=512, truncation=True)
    preds = classifier(X_test)

    y_pred = ["pos" if pred["label"] == "POSITIVE" else "neg" for pred in preds]
    get_scores(y_test, y_pred, "DistilBERT (not fine-tuned)")


def train_test_lstm(X_train, y_train, X_test, y_test):
    encoder = LabelBinarizer()

    Y_train_bin = encoder.fit_transform(y_train)
    Y_test_bin = encoder.transform(y_test)

    Y_train_bin = np.hstack((Y_train_bin, 1 - Y_train_bin))
    Y_test_bin = np.hstack((Y_test_bin, 1 - Y_test_bin))

    embeddings = read_embeddings(f"./data/glove.6B.200d.txt")

    vectorizer = tf.keras.layers.TextVectorization(
        standardize=None, output_sequence_length=100
    )

    text_ds = tf.data.Dataset.from_tensor_slices(X_train)

    vectorizer.adapt(text_ds)

    voc = vectorizer.get_vocabulary()
    emb_matrix = get_emb_matrix(voc, embeddings)
    X_train_vect = vectorizer(np.array([[s] for s in X_train])).numpy()
    X_test_vect = vectorizer(np.array([[s] for s in X_test])).numpy()

    model = tf.keras.models.Sequential()
    model.add(
        tf.keras.layers.Embedding(
            len(emb_matrix),
            200,
            embeddings_initializer=tf.keras.initializers.Constant(emb_matrix),
            trainable=False,
        )
    )

    model.add(tf.keras.layers.LSTM(units=512, activation="relu"))
    model.add(tf.keras.layers.Dropout(0.7))

    model.add(tf.keras.layers.Dense(input_dim=200, units=2, activation="softmax"))

    # Compile the model
    model.compile(
        loss="categorical_crossentropy",
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
        metrics=["accuracy"],
    )

    callbacks = get_callbacks(50)

    # Train the model
    model.fit(
        X_train_vect,
        Y_train_bin,
        batch_size=8,
        epochs=50,
        validation_split=0.1,
        callbacks=callbacks,
        verbose=1,
    )

    Y_pred = model.predict(X_test_vect)
    Y_pred = np.argmax(Y_pred, axis=1)
    Y_test = np.argmax(Y_test_bin, axis=1)

    get_scores(Y_test, Y_pred, "Best LSTM Model")


def train_test_distilbert(X_train, y_train, X_test, y_test):
    encoder = LabelBinarizer()

    Y_train_bin = encoder.fit_transform(y_train)
    Y_test_bin = encoder.transform(y_test)

    Y_train_bin = np.hstack((Y_train_bin, 1 - Y_train_bin))
    Y_test_bin = np.hstack((Y_test_bin, 1 - Y_test_bin))

    lm = "distilbert-base-uncased-finetuned-sst-2-english"
    tokenizer = AutoTokenizer.from_pretrained(lm)
    model = TFAutoModelForSequenceClassification.from_pretrained(lm)

    # Convert the data to data appropriate for the model
    tokens_train = tokenizer(
        X_train, padding=True, max_length=100, truncation=True, return_tensors="np"
    ).data
    tokens_test = tokenizer(
        X_test, padding=True, max_length=100, truncation=True, return_tensors="np"
    ).data

    # Initialize the optimizer, epochs and the loss function
    loss_function = tf.keras.losses.CategoricalCrossentropy(from_logits=True)
    optim = tf.keras.optimizers.Adam(learning_rate=5e-5)
    epochs = 3

    # Compile the model
    model.compile(loss=loss_function, optimizer=optim, metrics=["accuracy"])

    # Get the callbacks

    # Train the model
    model.fit(
        tokens_train, Y_train_bin, verbose=1, epochs=epochs, batch_size=8,
    )

    Y_pred = model.predict(tokens_test)
    Y_pred = np.argmax(Y_pred, axis=1)
    Y_test = np.argmax(Y_test_bin, axis=1)

    get_scores(Y_test, Y_pred, "DistilBERT - Fine-tuned")
