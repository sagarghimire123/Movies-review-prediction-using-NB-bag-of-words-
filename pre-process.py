

import os
import sys
import json

def count_frequencies(text):
    freq = {}
    for word in text:
        if word in freq:
            freq[word] += 1
        else:
            freq[word] = 1
    return freq


def ignore_unseen_words(words, vocabulary):
    return [word for word in words if word in vocabulary]

def remove_punctuation(text):

    punctuation_to_remove = {'"', '*', '+', '.', '/', '<', '>', '@'}
    new_text = ""
    for char in text:
        if char not in punctuation_to_remove:
            new_text += char.lower()
    return new_text.split()


def preprocess(directory, vocabul):
    feature_vectors = []
    for label in os.listdir(directory):
        folder = os.path.join(directory, label)
        if os.path.isdir(folder):
            for filename in os.listdir(folder):
                if filename.endswith(".txt"):
                    with open(os.path.join(folder, filename), "r") as f:
                        words = remove_punctuation(f.read())
                        words = ignore_unseen_words(words, vocabulary)
                        feature_vectors.append({label: count_frequencies(words)})
    output_name1 = "movie-review-BOWtrain.NB"
    with open(output_name1, "w") as output:
        for line in feature_vectors:
            output.write(json.dumps(line) + '\n')
directory1 = "/Users/ocn/Desktop/movie-review-HW2/aclImdb/train"
vocabulary = set([line.rstrip() for line in open('/Users/ocn/Desktop/movie-review-HW2/aclImdb/imdb.vocab')])
preprocess(directory1, vocabulary)

def preprocess_test_data(directory, vocabulary):
    feature_vectors = []
    for label in os.listdir(directory):
        folder = os.path.join(directory, label)
        if os.path.isdir(folder):
            for filename in os.listdir(folder):
                if filename.endswith(".txt"):
                    with open(os.path.join(folder, filename), "r") as f:
                        words = remove_punctuation(f.read())
                        words = ignore_unseen_words(words, vocabulary)
                        feature_vectors.append({label: count_frequencies(words)})
                        
    output_name2 = "movie-review-BOWtest.NB"
    with open(output_name2, "w") as output:
        for line in feature_vectors:
            output.write(json.dumps(line) + '\n')
directory2 = "/Users/ocn/Desktop/movie-review-HW2/aclImdb/test"
preprocess_test_data(directory2, vocabulary)





