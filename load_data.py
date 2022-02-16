import os
import json
import gzip
import csv
import codecs
import random


def write_json_to_file(path, data):
    with open(path, "w") as f:
        f.write(json.dumps(data))


def load_sentiment140_data():
    file = codecs.open(
        f"./data/training.1600000.processed.noemoticon.csv",
        "r",
        encoding="utf-8",
        errors="ignore",
    )

    csv_file = csv.DictReader(
        file, fieldnames=["target", "id", "date", "flag", "user", "text"]
    )

    data = []
    for row in csv_file:
        if row["target"] == "0":
            label = "neg"
        elif row["target"] == "2":
            continue
        else:
            label = "pos"

        data.append({"review": row["text"], "label": label})

    sentiment140_data = random.sample(data, 100000)
    write_json_to_file("./data/sentiment140.json", sentiment140_data)


def load_imdb_data():

    file = codecs.open(
        f"./data/imdb_master.csv", "r", encoding="utf-8", errors="ignore"
    )
    csv_file = csv.DictReader(file)

    test_data = []
    train_data = []

    for row in csv_file:
        if row["label"] == "unsup":
            continue
        data_type = row["type"]
        data = {"review": row["review"], "label": row["label"]}
        if data_type == "test":
            test_data.append(data)
        elif data_type == "train":
            train_data.append(data)

    write_json_to_file("./data/train_imdb.json", train_data)
    write_json_to_file("./data/test_imdb.json", test_data)


def load_amazon_data():
    data = []
    labels = []

    int_rating_to_label = {5.0: "pos", 4.0: "pos", 3.0: "neu", 2.0: "neg", 1.0: "neg"}
    for file in os.listdir("./data/"):
        if ".json.gz" in file:
            with gzip.open(f"/data/{file}") as f:
                for row in f:
                    json_row = json.loads(row)
                    if (
                        "reviewText" in json_row
                        and "overall" in json_row
                        and json_row["overall"] != 3.0
                    ):
                        data.append(
                            {
                                "review": json_row["reviewText"],
                                "label": int_rating_to_label[json_row["overall"]],
                            }
                        )

    write_json_to_file("./data/amazon.json", data)


if __name__ == "main":
    load_sentiment140_data()
    load_imdb_data()
    load_amazon_data()
