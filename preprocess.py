import json

from sklearn.model_selection import train_test_split
import nltk

nltk.download("punkt")
nltk.download("wordnet")
nltk.download("stopwords")
from string import punctuation
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize


def load_combined_data():
    with open("./data/amazon_reviews.json", "r") as f:
        amazon_data = json.load(f)

    with open("./data/train_imdb.json", "r") as f:
        train_imdb = json.load(f)
    with open("./data/test_imdb.json", "r") as f:
        test_imdb = json.load(f)

    with open("./data/sentiment140.json", "r") as f:
        sentiment140_data = json.load(f)

    combined_datasets = amazon_data + train_imdb + test_imdb + sentiment140_data

    reviews = [sample['review'] for sample in combined_datasets]
    labels = [sample['label'] for sample in combined_datasets]
    
    return reviews, labels


def remove_duplicates(reviews, labels):
    unique_reviews = []
    unique_labels = []

    for review, label in zip(reviews, labels):
        if review.lower() not in unique_reviews:
            unique_reviews.append(review.lower())
            unique_labels.append(label)

    return unique_reviews, unique_labels

def remove_outliers(reviews, labels):
    less_than_5 = [review for review in reviews if len(review) < 5]
    greater_than_3k = [review for review in reviews if len(review) > 3000]

    final_reviews = []
    final_labels = []
    for review, label in zip(reviews, labels):
        if review not in less_than_5 and review not in greater_than_3k:
            final_reviews.append(review)
            final_labels.append(label)

    return final_reviews, final_labels


def split_data(reviews, labels):
    X_train, X_test, y_train, y_test = train_test_split(reviews, labels, random_state=123, shuffle=True, test_size=0.15)
    
    return X_train, X_test, y_train, y_test


def convert_list_to_json(X, y):
    data = []
    for review, label in zip(X, y):
        data.append({
            'review': review,
            'label': label
        })
    return data

def create_stopwords():
    """Create stopwords from the nltk stopwords by removing the words that do influence the meaning in a sentence"""
    # Create a list of stopwords to remove
    # extract the original stopwords and remove the last words in that as they contain not (aren't, wouldn't, etc)
    stop_words_to_remove = ["not", "against", "down", "off", "over, no", "nor", "bottom"]
    original_stop_words = stopwords.words('english')[:144]
    
    # filter the stopwords
    return [word for word in original_stop_words if word not in stop_words_to_remove]


def preprocess_data(data):
    processed_data = []
    for sentence in data:
        tokens = word_tokenize(sentence)
        # Remove punctuation marks, do lemmatization and stemming, and remove stopwords
        processed_X = [x for x in tokens if x not in punctuation]
        processed_X = [x for x in tokens if x not in create_stopwords()]

        processed_data.append(processed_X)

    return processed_data

def write_json_to_file(file_name, data):
    with open(file_name, 'w') as f:
        json.dump(data, f)


if __name__ == "__main__":
    reviews, labels = load_combined_data()
    reviews, labels = remove_duplicates(reviews, labels)
    reviews, labels = remove_outliers(reviews, labels)
    reviews, labels = preprocess_data(reviews), labels
    X_train, X_test, y_train, y_test = split_data(reviews, labels)

    write_json_to_file("./data/train.json", convert_list_to_json(X_train, y_train))
    write_json_to_file("./data/test.json", convert_list_to_json(X_test, y_test))

    reviews, labels = preprocess_data(reviews), labels
    X_train, X_test, y_train, y_test = split_data(reviews, labels)

    write_json_to_file("./data/train_preprocessed.json", convert_list_to_json(X_train, y_train))
    write_json_to_file("./data/test_preprocesed.json", convert_list_to_json(X_test, y_test))