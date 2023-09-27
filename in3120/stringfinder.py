#!/usr/bin/python
# -*- coding: utf-8 -*-
from typing import Iterator, Dict, Any
from .tokenizer import Tokenizer
from .trie import Trie


class StringFinder:
    """
    Given a trie encoding a dictionary of strings, efficiently finds the subset of strings in the dictionary
    that are also present in a given text buffer. I.e., in a sense computes the "intersection" or "overlap"
    between the dictionary and the text buffer.

    Uses a trie-walk algorithm similar to the Aho-Corasick algorithm with some simplifications and some minor
    NLP extensions. The running time of this algorithm is virtually independent of the size of the dictionary,
    and linear in the length of the buffer we are searching in.

    The tokenizer we use when scanning the input buffer is assumed to be the same as the one that was used
    when adding strings to the trie.
    """

    def __init__(self, trie: Trie, tokenizer: Tokenizer):
        self.__trie = trie
        self.__tokenizer = tokenizer

    def scan(self, buffer: str) -> Iterator[Dict[str, Any]]:
        """
        Scans the given buffer and finds all dictionary entries in the trie that are also present in the
        buffer. We only consider matches that begin and end on token boundaries.

        The matching dictionary entries, if any, are yielded back to the client as dictionaries having the
        keys "match" (str) and "range" (Tuple[int, int]).

        In a serious application we'd add more lookup/evaluation features, e.g., support for prefix matching,
        support for leftmost-longest matching (instead of reporting all matches), and support for lemmatization
        or similar linguistic variations.
        """
        tokens = list(self.__tokenizer.tokens(buffer))
        matches = []
        for i, (token, (start, end)) in enumerate(tokens):
            counter = 1 # keeps count of added tokens in trie walk
            node = self.__trie.consume(token)
            # if there is a node with the consumed token
            while node:
                # if the node is final, i.e. an output node
                if node.is_final():
                    matches.append({"match": token, "range": (start, end)})
                # if the node is also a partial word, we can add the next token
                # this serves as the transition function
                if self.__trie.consume(token + ' '):
                    added_token, (added_start, added_end) = tokens[i + counter]
                    token = ' '.join([token, added_token])
                    end = added_end # adjusts the range
                    # if there is a node with the now extended token, the while loop continues
                    node = self.__trie.consume(token)
                    counter += 1
                # if there's not a node with the consumed token, the while loop stops and we move to the next token
                # this serves as the failure function
                else:
                    break
        sorted_matches = sorted(matches, key=lambda x: x['range'][1])
        yield from sorted_matches

