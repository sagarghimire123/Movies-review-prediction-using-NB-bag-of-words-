


import json
import math
class NaiveBayesClassifier:
    def __init__(self, train_data, vocabulary):
        self.train_data = train_data
        self.vocabulary = vocabulary
        self.classes = set(label for label, _ in train_data)
        self.priors = {}
        self.likelihoods = {label: {word: 0 for word in vocabulary} for label in self.classes}
        self.compute_priors()
        self.compute_likelihoods()
    def compute_priors(self):
        total_docs = len(self.train_data)
        for label in self.classes:
            docs_in_class = sum(1 for lbl, _ in self.train_data if lbl == label)
            self.priors[label] = docs_in_class / total_docs
    def compute_likelihoods(self):
        class_word_counts = {label: {word: 0 for word in self.vocabulary} for label in self.classes}
        class_total_words = {label: 0 for label in self.classes}
        for label, doc in self.train_data:
            for word, count in doc.items():
                class_word_counts[label][word] += count
                class_total_words[label] += count
        for label in self.classes:
            total_words = sum(class_word_counts[label].values())
            for word in self.vocabulary:
                self.likelihoods[label][word] = (class_word_counts[label][word] + 1) / (total_words + len(self.vocabulary))
    def predict(self, test_features):
        class_probs = {}
        for label in self.classes:
            class_probs[label] = math.log(self.priors[label])
            for word, count in test_features.items():
                if word in self.vocabulary:
                    class_probs[label] += count * math.log(self.likelihoods[label][word])

        return max(class_probs, key=class_probs.get)
    def calculate_accuracy(self, test_data):
        correct_predictions = 0
        total_predictions = len(test_data)
        for label, features in test_data:
            predicted_class = self.predict(features)
            if predicted_class == label:
                correct_predictions += 1
        return correct_predictions / total_predictions


def load_data(file_path):
    data = []
    with open(file_path, "r") as file:
        for line in file:
            vector = json.loads(line)
            label = list(vector.keys())[0]  
            features = vector[label]  
            data.append((label, features)) 
    return data

train_input_file_path = "/Users/ocn/desktop/NLP_assignment2/movie-review-BOWtrain.NB"
training_data = load_data(train_input_file_path)
vocabulary = set([line.rstrip() for line in open('/Users/ocn/Desktop/movie-review-HW2/aclImdb/imdb.vocab')])
classifier = NaiveBayesClassifier(training_data, vocabulary)
test_input_file_path = "/Users/ocn/desktop/NLP_assignment2/movie-review-BOWtest.NB"
test_data = load_data(test_input_file_path)

predicted_labels = [classifier.predict(features) for _, features in test_data]
actual_labels = [label for label, _ in test_data]
accuracy = classifier.calculate_accuracy(test_data)
accuracy





