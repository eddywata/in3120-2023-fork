#!/usr/bin/python
# -*- coding: utf-8 -*-

import itertools
from abc import ABC, abstractmethod
from collections import Counter
from typing import Iterable, Iterator, List
from .dictionary import InMemoryDictionary
from .normalizer import Normalizer
from .tokenizer import Tokenizer
from .corpus import Corpus
from .posting import Posting
from .postinglist import CompressedInMemoryPostingList, InMemoryPostingList, PostingList


class InvertedIndex(ABC):
    """
    Abstract base class for a simple inverted index.
    """

    def __getitem__(self, term: str) -> Iterator[Posting]:
        return self.get_postings_iterator(term)

    def __contains__(self, term: str) -> bool:
        return self.get_document_frequency(term) > 0

    @abstractmethod
    def get_terms(self, buffer: str) -> Iterator[str]:
        """
        Processes the given text buffer and returns an iterator that yields normalized
        terms as they are indexed. Both query strings and documents need to be
        identically processed.
        """
        pass

    @abstractmethod
    def get_postings_iterator(self, term: str) -> Iterator[Posting]:
        """
        Returns an iterator that can be used to iterate over the term's associated
        posting list. For out-of-vocabulary terms we associate empty posting lists.
        """
        pass

    @abstractmethod
    def get_document_frequency(self, term: str) -> int:
        """
        Returns the number of documents in the indexed corpus that contains the given term.
        """
        pass


class InMemoryInvertedIndex(InvertedIndex):
    """
    A simple in-memory implementation of an inverted index, suitable for small corpora.

    In a serious application we'd have configuration to allow for field-specific NLP,
    scale beyond current memory constraints, have a positional index, and so on.

    If index compression is enabled, only the posting lists are compressed. Dictionary
    compression is currently not supported.
    """

    def __init__(
        self,
        corpus: Corpus,
        fields: Iterable[str],
        normalizer: Normalizer,
        tokenizer: Tokenizer,
        compressed: bool = False,
    ):
        self.__corpus = corpus
        self.__normalizer = normalizer
        self.__tokenizer = tokenizer
        self.__posting_lists: List[PostingList] = []
        self.__dictionary = InMemoryDictionary()
        self.__build_index(fields, compressed)

    def __repr__(self):
        return str({term: self.__posting_lists[term_id] for (term, term_id) in self.__dictionary})

    def __build_index(self, fields: Iterable[str], compressed: bool) -> None:
        for doc in self.__corpus:
            line = " ".join([doc.get_field(field, None) for field in fields])
            doc_id = doc.get_document_id()
            canonicalized_line = self.__normalizer.canonicalize(line)
            strings = self.__tokenizer.strings(canonicalized_line)
            normalized_tokens = []
            for string in strings:
                normalized_token = self.__normalizer.normalize(string)
                normalized_tokens.append(normalized_token)

            counter = Counter(normalized_tokens)
            for term, count in counter.items():
                self.__dictionary.add_if_absent(term)
                posting = Posting(doc_id, count)

                term_id = self.__dictionary.get_term_id(term)

                # if the term is not new
                if len(self.__posting_lists) > term_id:
                    posting_list = self.__posting_lists[term_id]
                    posting_list.append_posting(posting)
                else:
                    posting_list = InMemoryPostingList()
                    self.__posting_lists.append(posting_list)
                    posting_list.append_posting(posting)

    def get_terms(self, buffer: str) -> Iterator[str]:
        terms_list = []
        canonicalized_buffer = self.__normalizer.canonicalize(buffer)
        terms = self.__tokenizer.strings(canonicalized_buffer)
        for term in terms:
            normalized_token = self.__normalizer.normalize(term)
            terms_list.append(normalized_token)
        return iter(terms_list)

    def get_postings_iterator(self, term: str) -> Iterator[Posting]:
        term_id = self.__dictionary.get_term_id(term)
        if not term_id:
            return iter([])
        return self.__posting_lists[term_id].get_iterator()

    def get_document_frequency(self, term: str) -> int:
        term_id = self.__dictionary.get_term_id(term)
        if not term_id:
            return 0
        return self.__posting_lists[term_id].get_length()
