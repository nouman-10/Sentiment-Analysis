import argparse


from model import *


def create_arg_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-m",
        "--model_name",
        default="nb",
        type=str,
        help="Model to train (af: afinn, nb: naive bayes, lr: logistic regression, svm: support vector machine, bert: distil-bert) (default: cnn)",
    )
    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = create_arg_parser()

    X_train, y_train, X_test, y_test = load_data(
        "./data/train.json", "./data/test.json"
    )

    if args.model_name == "af":
        test_afinn_model(X_train, y_train, X_test, y_test)

    elif args.model_name == "lr":
        train_test_logistic_reg(X_train, y_train, X_test, y_test)

    elif args.model_name == "svm":
        train_test_SVM(X_train, y_train, X_test, y_test)

    elif args.model_name == "nb":
        train_test_naive_bayes(X_train, y_train, X_test, y_test)

    else:
        train_test_distilbert(X_train, y_train, X_test, y_test)
