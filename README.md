# NLP
# PRACtica 1 :-Convert the text into tokens. Find the word frequency
import nltk
nltk.download('punkt')
sentence = """This tokenizer divides a text into a list of sentences by using an unsupervised algorithm to build a model for abbreviation words, collocations, and words that start sentences. It must be trained on a large collection of plaintext in the target language before it can be used."""
tokens = nltk.word_tokenize(sentence)
print("\n")
print(tokens)
from collections import Counter
word_freq = Counter(tokens)
print(word_freq)



# Practical 2 :- Find the synonym /antonym of a word using WordNet
import nltk
from nltk.corpus import wordnet
nltk.download('wordnet')
nltk.download('omw-1.4')

def find_synonyms_antonyms(word):
    synonyms = []
    antonyms = []

    for syn in wordnet.synsets(word):
        for lemma in syn.lemmas():

            synonyms.append(lemma.name())

            antonym_lemmas = lemma.antonyms()
            if antonym_lemmas:
                for antonym in antonym_lemmas:
                  antonyms.append(antonym.name())

    print(f"Synonyms of {word}: {set(synonyms)}")
    print(f"Antonyms of {word}: {set(antonyms)}")


# Practical 3:-Demonstrate a bigram / trigram language model

import nltk
nltk.download('punkt')
from nltk import ngrams
from nltk.tokenize import word_tokenize
sentence = "N-grams enhance language processing tasks."
tokens = word_tokenize(sentence)
bigrams = list(ngrams(tokens, 2))
trigrams = list(ngrams(tokens, 3))
print("Bigrams:", bigrams)
print("Trigrams:", trigrams)



# Practical 4:- Perform Lemmatization and Stemming. Identify parts-of Speech using Penn Treebank tag set.
import nltk
nltk.download('punkt')
from nltk.stem import PorterStemmer, WordNetLemmatizer
from nltk import pos_tag, word_tokenize
# Initialize stemmer and lemmatizer
stemmer = PorterStemmer()
lemmatizer = WordNetLemmatizer()
# Sample sentence
sentence = "The cats are running quickly."
tokens = word_tokenize(sentence)
# Perform stemming
stemmed_words = [stemmer.stem(word) for word in tokens]
print("Stemmed words:", stemmed_words)
# Perform lemmatization
lemmatized_words = [lemmatizer.lemmatize(word, pos='v') for word in tokens]
print("Lemmatized words:", lemmatized_words)
nltk.download('averaged_perceptron_tagger')
pos_tags = pos_tag(tokens)
print("POS tags:", pos_tags)
import nltk
from nltk.stem import WordNetLemmatizer, PorterStemmer
from nltk import pos_tag, word_tokenize
from nltk.corpus import wordnet
nltk.download('wordnet')
nltk.download('averaged_perceptron_tagger')
nltk.download('punkt')
text = "Perform Lemmatization and Stemming. Identify parts-of Speech using Penn Treebank tag set.
tokens = word_tokenize(text.lower())
lemmatizer = WordNetLemmatizer()
stemmer = PorterStemmer()
lemmatized_words = [lemmatizer.lemmatize(token) for token in tokens]
print("Lemmatized words:", lemmatized_words)
stemmed_words = [stemmer.stem(token) for token in tokens]
print("Stemmed words:", stemmed_words
)
pos_tags = pos_tag(tokens)
print("POS Tags:", pos_tags)





# Practical 5 :-Implement HMM for POS tagging. Build a Chunker

import nltk
from nltk.corpus import treebank
from nltk.tag import hmm
nltk.download('treebank')
train_data = treebank.tagged_sents()
trainer = hmm.HiddenMarkovModelTrainer()
hmm_tagger = trainer.train(train_data)
sentence = "The quick brown fox jumps over the lazy dog".split()
pos_tags = hmm_tagger.tag(sentence)
print("POS Tags:", pos_tags)





#Practical 6:-Implement Named Entity Recognizer.

# Download necessary NLTK data (if not already downloaded)
nltk.download('maxent_ne_chunker')
nltk.download('words')
nltk.download('averaged_perceptron_tagger')
nltk.download('punkt')
def named_entity_recognizer(text):
    # Tokenize the text
    tokens = nltk.word_tokenize(text)

    # Part-of-speech tagging
    pos_tags = nltk.pos_tag(tokens)

    # Named Entity Recognition using ne_chunk
    named_entities = nltk.ne_chunk(pos_tags)

    # Print the results (you can modify this to return the results in a different format)
    print(named_entities)


# Example usage
text = "Barack Obama was born in Honolulu, Hawaii."
named_entity_recognizer(text)





#Practical 7:- Implement Semantic Role Labeling (SRL) to Identify Named Entities

# prompt: Implement Semantic Role Labeling (SRL) to Identify Named Entities
import nltk
# Download necessary NLTK data (if not already downloaded)
nltk.download('maxent_ne_chunker')
nltk.download('words')
nltk.download('averaged_perceptron_tagger')
nltk.download('punkt')
nltk.download('conll2000')
from nltk.chunk import conlltags2tree, tree2conlltags
from pprint import pprint
def named_entity_recognizer(text):
    # Tokenize the text
    tokens = nltk.word_tokenize(text)

    # Part-of-speech tagging
    pos_tags = nltk.pos_tag(tokens)

    # Named Entity Recognition using ne_chunk
    named_entities = nltk.ne_chunk(pos_tags, binary=True) # Use binary=True for simpler output
    iob_tagged = tree2conlltags(named_entities)
    pprint(iob_tagged)

    # Print the results (you can modify this to return the results in a different format)
    #print(named_entities)

# Example usage
text = "Aditya was born in Honolulu, Hawaii. He studied at Columbia University."
named_entity_recognizer(text)







#Prictical 8:-Implement text classifier using logistic regression model

# prompt: Implement text classifier using logistic regression model

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# Sample data (replace with your actual data)
data = {'text': ['This is a positive sentence.', 'This is a negative sentence.', 'Another positive example.', 'A negative one.'],
        'label': [1, 0, 1, 0]}  # 1 for positive, 0 for negative
df = pd.DataFrame(data)

# Create TF-IDF features
vectorizer = TfidfVectorizer()
X = vectorizer.fit_transform(df['text'])
y = df['label']

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train a logistic regression model
model = LogisticRegression()
model.fit(X_train, y_train)

# Make predictions on the test set
y_pred = model.predict(X_test)

# Evaluate the model
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy}")

# Example prediction
new_text = ['This is a new positive sentence.']
new_text_vectorized = vectorizer.transform(new_text)
prediction = model.predict(new_text_vectorized)
print(f"Prediction for '{new_text[0]}': {prediction[0]}")





#Prictical 9:- Implement a movie reviews sentiment classifier

# prompt: Implement a movie reviews sentiment classifier

import nltk
import random
import numpy as np
import pandas as pd
from nltk.corpus import movie_reviews
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

# Download necessary NLTK data (if not already downloaded)
nltk.download('movie_reviews')
nltk.download('punkt')

# Prepare the data
documents = [(list(movie_reviews.words(fileid)), category)
              for category in movie_reviews.categories()
              for fileid in movie_reviews.fileids(category)]

random.shuffle(documents)

all_words = []
for w in movie_reviews.words():
    all_words.append(w.lower())

all_words = nltk.FreqDist(all_words)

word_features = list(all_words.keys())[:3000]

def find_features(document):
    words = set(document)
    features = {}
    for w in word_features:
        features[w] = (w in words)
    return features

featuresets = [(find_features(rev), category) for (rev, category) in documents]

training_set = featuresets[:1900]
testing_set = featuresets[1900:]

# Train the classifier
classifier = nltk.NaiveBayesClassifier.train(training_set)
print("Classifier accuracy percent:", (nltk.classify.accuracy(classifier, testing_set))*100)
classifier.show_most_informative_features(15)

# Example usage
example_text = "This movie was absolutely terrible. The acting was awful and the plot was confusing."
example_features = find_features(word_tokenize(example_text.lower()))
prediction = classifier.classify(example_features)
print(f"Prediction for '{example_text}': {prediction}")




#Prictical 10:-  Implement RNN for sequence labelling and show some output
import numpy as np

# Sample data (replace with your actual sequence labeling data)
sequences = [['The', 'quick', 'brown', 'fox'], ['jumps', 'over', 'the', 'lazy', 'dog']]
labels = [['DET', 'ADJ', 'ADJ', 'NOUN'], ['VERB', 'ADP', 'DET', 'ADJ', 'NOUN']]

# Create vocabulary and label dictionaries
word_to_index = {}
label_to_index = {}
index_to_label = {}
for seq, lab in zip(sequences, labels):
    for word in seq:
        if word not in word_to_index:
            word_to_index[word] = len(word_to_index)
    for label in lab:
        if label not in label_to_index:
            label_to_index[label] = len(label_to_index)
            index_to_label[len(label_to_index) - 1] = label

# Convert data to numerical representations
X = [[word_to_index[word] for word in seq] for seq in sequences]
y = [[label_to_index[label] for label in lab] for lab in labels]

# Pad sequences to ensure uniform length
max_len = max(len(seq) for seq in X)
X = [seq + [0] * (max_len - len(seq)) for seq in X]
y = [lab + [0] * (max_len - len(lab)) for lab in y]

# Adjust RNN weights and bias initialization
vocab_size = len(word_to_index)
hidden_size = 10  # Size of the hidden layer
output_size = len(label_to_index)

# Initialize weights and biases with adjusted shapes
np.random.seed(0)  # For reproducibility
weights = [
    np.random.rand(vocab_size, hidden_size),  # Input-to-hidden weights
    np.random.rand(hidden_size, hidden_size),  # Hidden-to-hidden weights
    np.random.rand(hidden_size, output_size)   # Hidden-to-output weights
]
bias = [np.random.rand(hidden_size), np.random.rand(output_size)]

# Define the RNN function with a simple training step
def simple_rnn_train(input_seqs, label_seqs, weights, bias, epochs=100, learning_rate=0.01):
    for epoch in range(epochs):
        total_loss = 0
        for input_seq, label_seq in zip(input_seqs, label_seqs):
            hidden_state = np.zeros(hidden_size)  # Initialize hidden state

            # Forward pass and loss calculation
            predicted_labels = []
            for t, word_index in enumerate(input_seq):
                input_vector = np.zeros(vocab_size)
                input_vector[word_index] = 1  # One-hot encoding
                hidden_state = np.tanh(np.dot(input_vector, weights[0]) + np.dot(hidden_state, weights[1]) + bias[0])

                # Predict label using the hidden state
                output_probs = np.dot(hidden_state, weights[2]) + bias[1]
                predicted_label_index = np.argmax(output_probs)
                predicted_labels.append(predicted_label_index)

                # Calculate loss (simple cross-entropy for this example)
                correct_label_index = label_seq[t]
                loss = -output_probs[correct_label_index] + np.log(np.sum(np.exp(output_probs)))
                total_loss += loss

                # Backpropagation and weight update
                # Output to hidden gradient
                output_delta = np.exp(output_probs) / np.sum(np.exp(output_probs))
                output_delta[correct_label_index] -= 1
                weights[2] -= learning_rate * np.outer(hidden_state, output_delta)
                bias[1] -= learning_rate * output_delta

                # Hidden to hidden gradient
                hidden_delta = (1 - hidden_state ** 2) * np.dot(weights[2], output_delta)
                weights[1] -= learning_rate * np.outer(hidden_state, hidden_delta)
                weights[0] -= learning_rate * np.outer(input_vector, hidden_delta)
                bias[0] -= learning_rate * hidden_delta

        if epoch % 10 == 0:
            print(f'Epoch {epoch}, Loss: {total_loss:.4f}')

    return weights, bias

# Train the RNN
weights, bias = simple_rnn_train(X, y, weights, bias, epochs=100, learning_rate=0.01)

# Predict labels for a sequence after training
def simple_rnn_predict(input_seq, weights, bias):
    hidden_state = np.zeros(hidden_size)  # Initialize hidden state

    outputs = []
    for word_index in input_seq:
        input_vector = np.zeros(vocab_size)
        input_vector[word_index] = 1  # One-hot encoding
        hidden_state = np.tanh(np.dot(input_vector, weights[0]) + np.dot(hidden_state, weights[1]) + bias[0])

        # Predict label using the hidden state
        output_probs = np.dot(hidden_state, weights[2]) + bias[1]
        predicted_label_index = np.argmax(output_probs)
        outputs.append(predicted_label_index)

    return outputs

# Test the RNN with predictions
predicted_labels = simple_rnn_predict(X[0], weights, bias)
print(predicted_labels)  # Output as indexes
predicted_labels_text = [index_to_label[pred] for pred in predicted_labels]
print(predicted_labels_text)






