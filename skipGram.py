from __future__ import division
import argparse
import pandas as pd

# useful stuff
import numpy as np
from scipy.special import expit
from sklearn.preprocessing import normalize
import re
import json

__authors__ = ['Louis_Veillon, Quentin_Boutoille-Blois']
__emails__ = ['b00727589@essec.edu', 'b00527749@essec.edu']


def text2sentences(path, stopwords):
 # feel free to make a better tokenization/pre-processing
    sentences = []
    with open(path) as f:
        for l in f:
            sentences.append(clean_sentence(l, stopwords))
    return sentences


def loadPairs(path):
    data = pd.read_csv(path, delimiter='\t')
    pairs = zip(data['word1'], data['word2'], data['similarity'])
    return pairs


def loadStopwords(path):
    # import stopwords file
    stopwords_file = open(path, 'r')
    stopwords = []
    for word in stopwords_file:
        stopwords.append(word.strip('\n'))

    return stopwords


def clean_sentence(sentence, stopwords):
    rx = re.compile('\W+')
    sentence = str(sentence).lower().split()
    sentence = [rx.sub(' ', i).strip() for i in sentence if i not in stopwords and rx.sub(' ', i).strip() != '']
    return sentence


class mySkipGram:

    def __init__(self, sentences, nEmbed=10, negativeRate=10, winSize=5, minCount=2):

        # winSize: Size of th window
        # minCount : minimum times word appears

        self.winSize = winSize
        self.minCount = minCount
        self.sentences = sentences
        self.nEmbed = int(nEmbed)
        self.negativeRate = negativeRate
        self.get_vocabulary(minCount)

        self.weights1_file = 'weight2.csv'
        self.weights2_file = 'weight1.csv'
        self.vocab_file = 'save_voca.csv'

    def get_vocabulary(self, minCount):

        # Mincount is the minimal counting value from which the word is included in vocabulary

        self.vocabulary = {}
        self.vocabulary_filtered = {}

        for sentence in self.sentences:

            for word in sentence:

                if word not in self.vocabulary:
                    self.vocabulary[word] = 1
                else:
                    self.vocabulary[word] += 1

        for word, value in self.vocabulary.items():
            if value > minCount:
                self.vocabulary_filtered[word] = value

        self.vocabulary = self.vocabulary_filtered
        self.length_vocabulary = len(self.vocabulary)
        self.vocabulary_list = list(self.vocabulary)

        return self.vocabulary
    """
    def word_to_vec(self, word):

        # transform a word into a hot vector

        word_vec = np.zeros(self.length_vocabulary)
        word_vec[vocabulary_list.index('word')] = 1
        return word_vec
    """

    def generate_D(self):

        # generate_D is  a function to generate true pairs Word, countext dataset

        self.Dictionary_D = {}

        for sentence in self.sentences:
            for word in sentence:
                position = sentence.index(word)

                if word in self.vocabulary:           # if the word belongs to the filtered vocabulary list (appears more than 3 times)

                    if word not in self.Dictionary_D:  # if the word is not a key yet
                        self.Dictionary_D[word] = []

                    for context_word in sentence:
                        if context_word in self.vocabulary:       # if the word belongs to the filtered vocabulary list
                            if context_word not in self.Dictionary_D[word]:  # if the word does not belong to context word list of the word yet
                                pos_context_word = sentence.index(context_word)

                                if np.abs(pos_context_word - position) <= int(self.winSize / 2) and np.abs(pos_context_word - position) > 0:
                                    self.Dictionary_D[word].append(context_word)

    def generate_D_prime(self):

        # generate_D_prime is a function generating false pairs (Word,context).
        # to choose a false context word we pick up randomly a word form the dictionary

        self.Dictionary_D_prime = {}

        # generate the negative sampling probability based on word frequencies
        count_list = []  # initialize a list of the word frequencies
        for word in self.vocabulary_list_new:  # loop over the vocabulary key
            count_list.append(self.vocabulary[word])  # add each word frequency to a list

        frequencies = np.asarray(count_list)  # get the word frequency into a np array
        proba = np.power(frequencies, 3 / 4)  # power to 3/4
        proba = proba / np.sum(proba)  # divide each powered frequency by the sum of all

        for sentence in self.sentences:
            for word in sentence:

                word_context_list = np.random.choice(self.vocabulary_list_new, size=self.negativeRate, p=proba)

                if word in self.vocabulary_list_new:

                    if word not in self.Dictionary_D_prime:
                        self.Dictionary_D_prime[word] = []
                    for word_context in word_context_list:
                        self.Dictionary_D_prime[word].append(word_context)

    def sigmoid(self, z):
        return expit(z)

    def train(self, stepsize, epochs):

        print("Training started...")
        # create D, the true Data Set (word, context)
        self.generate_D()

        # remove word from the vocabulary having no context word
        self.vocabulary_list_new = []
        for index_word, word in enumerate(self.vocabulary_list):
            if len(self.Dictionary_D[word]) != 0:
                self.vocabulary_list_new.append(word)

        self.length_vocabulary_new = len(self.vocabulary_list_new)

        # weights W_w
        self.W_1 = np.random.rand(self.length_vocabulary_new, self.nEmbed)

        # weights W_c
        self.W_2 = np.random.rand(self.length_vocabulary_new, self.nEmbed)
        # intermediary variable
        self.W_2_ = np.random.rand(self.length_vocabulary_new, self.nEmbed)

        self.generate_D_prime()

        for ep in range(epochs):

            print("Epoch: " + str(ep) + "/" + str(epochs))
            for index_word, word in enumerate(self.vocabulary_list_new):
                for word_context in self.Dictionary_D[word]:

                    index_word_context = self.vocabulary_list_new.index(word_context)

                    word_set = [(index_word, 1)] + [(self.vocabulary_list_new.index(word_neg), 0) for word_neg in self.Dictionary_D_prime[word]]

                    for wor, label in word_set:
                        # update W_2_
                        self.W_2_[wor, :] += stepsize * (label - self.sigmoid(np.dot(self.W_1[index_word_context, :], self.W_2[wor, :]))) * self.W_1[index_word_context, :]
                        # update W_1
                        self.W_1[index_word_context, :] += stepsize * (label - self.sigmoid(np.dot(self.W_1[index_word_context, :], self.W_2[wor, :]))) * self.W_2[wor, :]
                        # update W_2
                        self.W_2[wor, :] = self.W_2_[wor, :]

        print("Training completed")

    def save(self, path):

        var = self.__dict__.copy()

        # changing some types in order to get them into a json
        #var['vocabulary'] = var['vocabulary'].to_json()

        var['vocabulary'] = json.dumps(var['vocabulary'])
        var['W_1'] = var['W_1'].tolist()
        var['W_2'] = var['W_2'].tolist()

        # writing vars
        with open(path, 'w', encoding="utf-8") as file:
            file.write(json.dumps(var))
        file.close()

        print("The model has been saved successfully")

        pass

    def similarity(self, word1, word2):

        index_word1 = self.vocabulary_list_new.index(word1)
        index_word2 = self.vocabulary_list_new.index(word2)

        """ to compute the similarity between two words, we compute the formula v1.v2/(||v1||*||v2||)
        where v1 and v2 probability vector , (output vector).
        """

        return np.sum(np.multiply(self.sigmoid(np.dot(self.W_2, self.W_1[index_word1, :])), self.sigmoid(np.dot(self.W_2, self.W_1[index_word2, :])))) / (np.linalg.norm(self.sigmoid(np.dot(self.W_2, self.W_1[index_word1, :]))) * np.linalg.norm(self.sigmoid(np.dot(self.W_2, self.W_1[index_word2, :]))))
        """
            computes similiarity between the two words. unknown words are mapped to one common vector
        :param word1:
        :param word2:
        :return: a float \in [0,1] indicating the similarity (the higher the more similar)
        """

    @staticmethod
    def load(path):

        with open(path, 'r', encoding="utf-8") as file:
            var = json.load(file)
        file.close
        # instantiating without calling __init__ constructor
        new_skip_gram = mySkipGram.__new__(mySkipGram)

        # changing some var into their normal type
        var['vocabulary'] = pd.read_json(var['vocabulary'], typ='series', orient='records')
        var['W_1'] = np.array(var['W_1']).reshape(var["length_vocabulary"], var["nEmbed"])
        var['W_é'] = np.array(var['W_2']).reshape(var["nEmbed"], var["length_vocabulary"])

        # setting attributes to the new instance
        for attribute, attribute_value in var.items():
            setattr(new_skip_gram, attribute, attribute_value)
        print("The model has been loaded successfully")
        return new_skip_gram
        pass


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--text', help='path containing training data', required=True)
    parser.add_argument('--model', help='path to store/read model (when training/testing)', required=True)
    parser.add_argument('--test', help='enters test mode', action='store_true')

    opts = parser.parse_args()

    if not opts.test:
        sentences = text2sentences(opts.text)
        sg = mySkipGram(sentences)
        sg.train()
        sg.save(opts.model)

    else:
        pairs = loadPairs(opts.text)

        sg = mySkipGram.load(opts.model)
        for a, b, _ in pairs:
            print(sg.similarity(a, b))
