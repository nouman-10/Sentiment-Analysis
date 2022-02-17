# ML and DL algorithms for Sentiment Analysis

### Table of Contents

1. [Installation](#installation)
2. [Project Motivation](#motivation)
3. [File Descriptions](#files)

## Installation <a name="installation"></a>

The libraries required for the successful execution of this code are mentioned in requirements.txt. In order to install all the libraries:
`pip install -r requirements.txt`

## Project Motivation<a name="motivation"></a>


In this project, the problem of sentiment analysis tackled. The data contains text from various resources

Data Resources:
- Sentiment140: https://www.kaggle.com/kazanova/sentiment140
- IMDB: https://www.kaggle.com/lakshmi25npathi/imdb-dataset-of-50k-movie-reviews
- Amazon: https://nijianmo.github.io/amazon/index.html (Specific 5-core: fashion, beauty, appliances, gift cards, industrial, scientific, luxury beauty, magazine subscriptions, and software)
- Glove Embeddings: Download from: https://nlp.stanford.edu/projects/glove/
Various Machine Learning and Deep Learning algorithms are compared such as Logistic regression, Support Vector Machines, Naive Bayes, LSTMs, and BERT. 
## File Descriptions <a name="files"></a>

- `load_data.py` contains code to load data from the downloaded raw files. To run the file:

```python load_data.py```

- `preprocess.py` contains code to preprocess the loaded data. To run the file:
```python preprocess.py```

- `model.py` contains code for modeling different ML and DL algorithms

- `main.py` contains code to run the modeling for specific algorithm. For instance to run the file for  Naive bayes:

```python main.py -m nb```

The different models are:
- af: Afinn Lexicon
- lr: logistic regression
- svm: support vector machine
- nb: Naive Bayes
- lstm: LSTMs
- bert: BERT


