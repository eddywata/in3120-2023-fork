#!/usr/bin/python
# -*- coding: utf-8 -*-

from typing import Any, Dict, Iterator, Iterable, Tuple, List
from collections import defaultdict
from .corpus import Corpus
from .normalizer import Normalizer
from .tokenizer import Tokenizer
from .sieve import Sieve

class SuffixArray:
    """
    A simple suffix array implementation. Allows us to conduct efficient substring searches.
    The prefix of a suffix is an infix!

    In a serious application we'd make use of least common prefixes (LCPs), pay more attention
    to memory usage, and add more lookup/evaluation features.
    """

    def __init__(self, corpus: Corpus, fields: Iterable[str], normalizer: Normalizer, tokenizer: Tokenizer):
        self.__corpus = corpus
        self.__normalizer = normalizer
        self.__tokenizer = tokenizer
        self.__haystack: List[Tuple[int, str]] = []  # The (<document identifier>, <searchable content>) pairs.
        self.__suffixes: List[Tuple[int, int]] = []  # The sorted (<haystack index>, <start offset>) pairs.
        self.__build_suffix_array(fields)  # Construct the haystack and the suffix array itself.

    def __build_suffix_array(self, fields: Iterable[str]) -> None:
        """
        Builds a simple suffix array from the set of named fields in the document collection.
        The suffix array allows us to search across all named fields in one go.
        """
        for doc in self.__corpus:
            text = " ".join([doc.get_field(field, None) for field in fields])
            doc_id = doc.get_document_id()
            normalized_text = self.__normalize(text)
            self.__haystack.append((doc_id, normalized_text))

            text_list = normalized_text.split()
            current_offset = 0
            for word in text_list:
                word_offset_index = normalized_text.find(word, current_offset)
                self.__suffixes.append((doc_id, word_offset_index))
                current_offset = word_offset_index + len(word)

        # sorts the suffixes alphabetically
        self.__suffixes.sort(key=lambda suffix_tuple: self.__find_suffix_string(suffix_tuple, exact=True))

    def __find_suffix_string(self, suffix_offset: Tuple[int, int], exact: bool=False) -> str:
        """
        Takes a tuple from self.__suffixes and returns the string from self.__haystack which
        it is meant to represent. Can return suffix plus rest of string or exact suffix.
        :param suffix_offset: Tuple consisting of doc_id and start offset
        :param exact: Turn on/off end of suffix search for exact suffix, default off
        :return: The actual string from the haystack
        """
        doc_id = suffix_offset[0]
        doc = self.__haystack[doc_id]
        haystack_string = doc[1][suffix_offset[1]:]
        if exact:
            end_index = haystack_string.find(' ')
            if end_index != -1:
                return haystack_string[:end_index]
            else:
                return haystack_string
        else:
            return haystack_string

    def __normalize(self, buffer: str) -> str:
        """
        Produces a normalized version of the given string. Both queries and documents need to be
        identically processed for lookups to succeed.
        """
        canonicalized_buffer = self.__normalizer.canonicalize(buffer)
        strings = self.__tokenizer.strings(canonicalized_buffer)
        normalized_tokens = []
        for string in strings:
            normalized_token = self.__normalizer.normalize(string)
            normalized_tokens.append(normalized_token)
        return " ".join(normalized_tokens)

    def __binary_search(self, needle: str) -> int:
        """
        Does a binary search for a given normalized query (the needle) in the suffix array (the haystack).
        Returns the position in the suffix array where the normalized query is either found, or, if not found,
        should have been inserted.

        Kind of silly to roll our own binary search instead of using the bisect module, but seems needed
        prior to Python 3.10 due to how we represent the suffixes via (index, offset) tuples. Version 3.10
        added support for specifying a key.
        """
        needle = self.__normalize(needle)
        low = 0
        high = len(self.__suffixes)
        while low <= high:
            i = (low + high) // 2
            try:
                current_suffix = self.__suffixes[i]
            except IndexError:
                # if the index is out of range, the query should have been inserted at the end
                return high - 1
            string = self.__find_suffix_string(current_suffix)
            if string == needle:
                return i
            elif string < needle:
                low = i + 1
            elif string > needle:
                high = i - 1
        return low

    def evaluate(self, query: str, options: dict) -> Iterator[Dict[str, Any]]:
        """
        Evaluates the given query, doing a "phrase prefix search".  E.g., for a supplied query phrase like
        "to the be", we return documents that contain phrases like "to the bearnaise", "to the best",
        "to the behemoth", and so on. I.e., we require that the query phrase starts on a token boundary in the
        document, but it doesn't necessarily have to end on one.

        The matching documents are ranked according to how many times the query substring occurs in the document,
        and only the "best" matches are yielded back to the client. Ties are resolved arbitrarily.

        The client can supply a dictionary of options that controls this query evaluation process: The maximum
        number of documents to return to the client is controlled via the "hit_count" (int) option.

        The results yielded back to the client are dictionaries having the keys "score" (int) and
        "document" (Document).
        """
        # if the query is empty, return no results rather than all documents
        if not query:
            return None

        query = self.__normalize(query)
        hit_count = options.get("hit_count", float("inf"))
        suffix_index = self.__binary_search(query)
        suffix_tuple = self.__suffixes[suffix_index]

        doc_ids = set()
        doc_id = suffix_tuple[0]

        poss_matches = defaultdict(list)  # dictionary with key:doc_id and value:possible match
        poss_match = self.__find_suffix_string(suffix_tuple)
        exact_poss_match = self.__find_suffix_string(suffix_tuple, exact=True)
        poss_matches[doc_id].append(poss_match)
        doc_ids.add(doc_id)

        # finds potential earlier matches,
        # iterates backwards from original possible match returned from binary search
        for earlier_tuple in self.__suffixes[suffix_index-1::-1]:
            earlier_string = self.__find_suffix_string(earlier_tuple, exact=True)
            if query < earlier_string:
                doc_id = earlier_tuple[0]
                earlier_poss_match = self.__find_suffix_string(earlier_tuple)
                poss_matches[doc_id].append(earlier_poss_match)
                doc_ids.add(doc_id)
            else:
                break
        # finds potential later matches,
        # iterates from original possible match returned from binary search
        for later_tuple in self.__suffixes[suffix_index+1:]:
            doc_id = later_tuple[0]
            later_string = self.__find_suffix_string(later_tuple, exact=True)
            later_poss_match = self.__find_suffix_string(later_tuple)
            # if the query is shorter than the later string, adjust length for comparison
            if len(later_string) > len(query):
                partial_later_string = later_string[:len(query)]
                if partial_later_string == query:
                    poss_matches[doc_id].append(later_poss_match)
                    doc_ids.add(doc_id)
            elif later_string == exact_poss_match:
                poss_matches[doc_id].append(later_poss_match)
                doc_ids.add(doc_id)
            else:
                break

        doc_hits = []

        for doc_id in doc_ids:
            doc_hits_dict = {'score': len(poss_matches[doc_id]), 'document': self.__corpus.get_document(doc_id)}
            for match in poss_matches[doc_id]:
                for query_char, match_char in zip(query, match):
                    if query_char != match_char:
                        doc_hits_dict['score'] -= 1
                        break

            if doc_hits_dict['score'] > 0:
                doc_hits.append(doc_hits_dict)

        # the sieve decides the winners
        # sieve doesn't allow dicts or Documents, so this passes through the score and document_id,
        # and then retrieves this function's correct return format after everything has been sifted
        sieve = Sieve(hit_count)
        for hit in doc_hits:
            score = hit['score']
            sieve.sift(score, hit['document'].document_id)
        for winner in sieve.winners():
            yield {'score': winner[0], 'document': self.__corpus.get_document(winner[1])}
