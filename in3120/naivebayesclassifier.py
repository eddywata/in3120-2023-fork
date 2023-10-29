#!/usr/bin/python
# -*- coding: utf-8 -*-

from typing import Any, Dict, Iterable, Iterator
from .dictionary import InMemoryDictionary
from .normalizer import Normalizer
from .tokenizer import Tokenizer
from .corpus import Corpus
import itertools
from collections import Counter


class NaiveBayesClassifier:
    """
    Defines a multinomial naive Bayes text classifier.
    """

    def __init__(self, training_set: Dict[str, Corpus], fields: Iterable[str],
                 normalizer: Normalizer, tokenizer: Tokenizer):
        """
        Constructor. Trains the classifier from the named fields in the documents in
        the given training set.
        """
        # Used for breaking the text up into discrete classification features.
        self.__normalizer = normalizer
        self.__tokenizer = tokenizer

        # The vocabulary we've seen during training.
        self.__vocabulary = InMemoryDictionary()

        # Maps a category c to the prior probability Pr(c).
        self.__priors: Dict[str, float] = {}

        # Maps a category c and a term t to the conditional probability Pr(t | c).
        self.__conditionals: Dict[str, Dict[str, float]] = {}

        # Maps a category c to the denominator used when doing Laplace smoothing.
        self.__denominators: Dict[str, int] = {}

        # Train the classifier, i.e., estimate all probabilities.
        self.__compute_priors(training_set)
        self.__compute_vocabulary(training_set, fields)
        self.__compute_posteriors(training_set, fields)

    def __compute_priors(self, training_set):
        """
        Estimates all prior probabilities needed for the naive Bayes classifier.
        """
        total_docs_size = sum(corpus.size() for corpus in training_set.values())
        for category, corpus in training_set.items():
            self.__priors[category] = corpus.size() / total_docs_size

    def __compute_vocabulary(self, training_set, fields):
        """
        Builds up the overall vocabulary as seen in the training set.
        """
        for _, corpus in training_set.items():
            for doc in corpus:
                terms = itertools.chain.from_iterable(self.__get_terms(doc.get_field(f, "")) for f in fields)
                for term in terms:
                    self.__vocabulary.add_if_absent(term)

    def __compute_posteriors(self, training_set, fields):
        """
        Estimates all conditional probabilities needed for the naive Bayes classifier.
        """
        for category, corpus in training_set.items():
            term_frequencies = Counter()
            for doc in corpus:
                terms = itertools.chain.from_iterable(self.__get_terms(doc.get_field(f, "")) for f in fields)
                term_frequencies.update(terms)
            # length of category text plus length of vocabulary is added as denominator
            self.__denominators[category] = sum(term_frequencies.values()) + len(self.__vocabulary)

            for term, freq in term_frequencies.items():
                # TODO: rm print
                # print(f'P({term}|{category}) = {freq}+1/{sum(term_frequencies.values())}+{len(self.__vocabulary)}')
                # conditional probabilities are calculated by term frequency within category plus Laplace smoothing,
                # divided by the denominator
                self.__conditionals[category] = {term: float()}
                self.__conditionals[category][term] = ((freq + 1) / (self.__denominators[category]))
                print(self.__conditionals[category][term])

    def __get_terms(self, buffer):
        """
        Processes the given text buffer and returns the sequence of normalized
        terms as they appear. Both the documents in the training set and the buffers
        we classify need to be identically processed.
        """
        tokens = self.__tokenizer.strings(self.__normalizer.canonicalize(buffer))
        return (self.__normalizer.normalize(t) for t in tokens)

    def classify(self, buffer: str) -> Iterator[Dict[str, Any]]:
        """
        Classifies the given buffer according to the multinomial naive Bayes rule. The computed (score, category) pairs
        are emitted back to the client via the supplied callback sorted according to the scores. The reported scores
        are log-probabilities, to minimize numerical underflow issues. Logarithms are base e.

        The results yielded back to the client are dictionaries having the keys "score" (float) and
        "category" (str).
        """
        ...
