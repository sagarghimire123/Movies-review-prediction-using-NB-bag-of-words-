



import pandas as pd 
import numpy as np
import math

train_data = [
    ('comedy', {'fun': 1, 'couple': 1, 'love': 2}),
    ('action', {'fast': 1, 'furious': 1, 'shoot': 2}),
    ('comedy', {'couple': 1, 'fly': 1, 'fast': 1, 'fun': 2}),
    ('action', {'furious': 1, 'shoot': 2, 'fun': 1}),
    ('action', {'fly': 1, 'fast': 1, 'shoot': 1, 'love': 1})
]


test_features = {'fast': 1, 'couple': 1, 'shoot': 1, 'fly': 1}
vocabulary = {'fun', 'couple', 'love', 'fast', 'furious', 'shoot', 'fly'}

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

        return class_probs

    def calculate_accuracy(self, test_data):
        correct_predictions = 0
        total_predictions = len(test_data)
        for label, features in test_data:
         
            predicted_class = max(self.predict(features), key=self.predict(features).get)
        
            if predicted_class == label:
                correct_predictions += 1
        # Compute accuracy
        return correct_predictions / total_predictions


    def calculate_prior_class_prob(self, label):
        return self.priors[label]


    def calculate_log_prob(self, label):
        return math.log(self.priors[label])  


# Create and train the classifier
classifier = NaiveBayesClassifier(train_data, vocabulary)

# Prior class probabilities
print("Prior class probabilities:")
for label in classifier.classes:
    prior = classifier.calculate_prior_class_prob(label)
    print("Prior class probability ({}): {:.4f}".format(label, prior))

# Log probabilities for test data
print("\nLog probabilities for test data:")
class_probs = classifier.predict(test_features)
for label, prob in class_probs.items():
    print("Probability of class", label, ":", prob)






