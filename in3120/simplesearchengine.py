#!/usr/bin/python
# -*- coding: utf-8 -*-

from collections import Counter
from typing import Iterator, Dict, Any
from .sieve import Sieve
from .ranker import Ranker
from .corpus import Corpus
from .invertedindex import InvertedIndex


class SimpleSearchEngine:
    """
    Realizes a simple query evaluator that efficiently performs N-of-M matching over an inverted index.
    I.e., if the query contains M unique query terms, each document in the result set should contain at
    least N of these m terms. For example, 2-of-3 matching over the query 'orange apple banana' would be
    logically equivalent to the following predicate:

       (orange AND apple) OR (orange AND banana) OR (apple AND banana)
       
    Note that N-of-M matching can be viewed as a type of "soft AND" evaluation, where the degree of match
    can be smoothly controlled to mimic either an OR evaluation (1-of-M), or an AND evaluation (M-of-M),
    or something in between.

    The evaluator uses the client-supplied ratio T = N/M as a parameter as specified by the client on a
    per query basis. For example, for the query 'john paul george ringo' we have M = 4 and a specified
    threshold of T = 0.7 would imply that at least 3 of the 4 query terms have to be present in a matching
    document.
    """

    def __init__(self, corpus: Corpus, inverted_index: InvertedIndex):
        self.__corpus = corpus
        self.__inverted_index = inverted_index

    def evaluate(self, query: str, options: dict, ranker: Ranker) -> Iterator[Dict[str, Any]]:
        """
        Evaluates the given query, doing N-out-of-M ranked retrieval. I.e., for a supplied query having M
        unique terms, a document is considered to be a match if it contains at least N <= M of those terms.

        The matching documents, if any, are ranked by the supplied ranker, and only the "best" matches are yielded
        back to the client as dictionaries having the keys "score" (float) and "document" (Document).

        The client can supply a dictionary of options that controls the query evaluation process: The value of
        N is inferred from the query via the "match_threshold" (float) option, and the maximum number of documents
        to return to the client is controlled via the "hit_count" (int) option.
        """
        inverted_index = self.__inverted_index
        query = list(inverted_index.get_terms(query))

        m = len(query)
        n = max(1, min(m, int(options['match_threshold'] * m)))

        # collect posting lists for each term in query
        postings_lists = []
        for term in query:
            print(term)
            postings = inverted_index.get_postings_iterator(term)
            postings_lists.append(postings)

        sieve = Sieve(options['hit_count'])

        # initial postings for all terms in query, will function as cursors throughout iteration
        all_cursors = [next(p, None) for p in postings_lists]

        print(all_cursors)
        while all_cursors and len(all_cursors) >= n:
            min_doc_id = min([cursor.document_id for cursor in all_cursors])

            # the lowest document IDs mentioned in non-exhausted posting lists
            frontier = [i for i, cursor in enumerate(all_cursors) if cursor is not None and cursor.document_id == min_doc_id]

            if len(frontier) >= n:
                scores = [cursor.term_frequency for cursor in (all_cursors[i] for i in frontier)]
                for i, score in zip(frontier, scores):
                    sieve.sift(score, all_cursors[i].document_id)
            for i in frontier:
                cursor = all_cursors[i] # TODO: unnecessary?
                next_posting = next(postings_lists[i], None)
                all_cursors[i] = next_posting

            all_cursors = [cursor for cursor in all_cursors if cursor is not None]

        # TODO: the loop may be alright. next is sieve winners, ranking and yielding?


        #
        # non_exhausted_lists = [i for i, cursor in enumerate(all_cursors) if cursor is not None]
        #
        # while non_exhausted_lists:
        #     min_doc_id = min([cursor.document_id for cursor in non_exhausted_lists])
        #     frontier = [cursor for cursor in non_exhausted_lists if cursor.document_id == min_doc_id]
        #
        #     if len(frontier) < n:
        #         break
        #
        #     doc_id = all_cursors[frontier[0]][0]
        #

        # while all_cursors and len(all_cursors) >= n:
        #     print(all_cursors is None)
        #
        #     min_doc_id = min([cursor.document_id for cursor in all_cursors if cursor is not None])
        #     frontier = [i for i, cursor in enumerate(all_cursors) if cursor is not None if cursor.document_id == min_doc_id]
        #
        #     if len(frontier) >= n:
        #         scores = [cursor.term_frequency for cursor in (all_cursors[i] for i in frontier)]
        #         for i, score in zip(frontier, scores):
        #             sieve.sift(score, all_cursors[i].document_id)
        #
        #     for i in frontier:
        #         cursor = all_cursors[i]
        #         next_posting = next(posting_lists[i], None)
        #         all_cursors[i] = next_posting


        # sieve = Sieve(options['hit_count'])
        #
        # top_results = sieve.winners()
        #
        # top_results = sorted(top_results, key=lambda x: x[0], reverse=True)
        #
        # for score, cursor in top_results:
        #     document = self.__corpus.get_document(cursor)
        #     yield {'score': score, 'document': document}