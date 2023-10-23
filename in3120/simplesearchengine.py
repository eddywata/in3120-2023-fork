#!/usr/bin/python
# -*- coding: utf-8 -*-

from collections import Counter
from typing import Iterator, Dict, Any
from .sieve import Sieve
from .ranker import Ranker, SimpleRanker
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
        # turning query into list
        query = list(inverted_index.get_terms(query))
        # counting query terms for multiplicity
        query_counter = Counter(query)
        # keeping only unique terms from query
        query = list(dict.fromkeys(query))

        m = len(query)
        n = max(1, min(m, int(options['match_threshold'] * m)))

        # collect posting lists for each term in query
        posting_lists = []
        for term in query:
            posting_list = inverted_index.get_postings_iterator(term)
            posting_lists.append([posting_list, term])  # the term is also kept, for later reference

        sieve = Sieve(options['hit_count'])

        # initial postings for all terms in query, will function as cursors throughout search
        all_cursors = [[next(posting_list, None), term] for posting_list, term in posting_lists]
        # remove potential empty posting lists
        all_cursors = [cursor for cursor in all_cursors if cursor[0] is not None]
        # keep going as long as there are at least n posting lists
        while all_cursors and len([cursor for cursor, _ in all_cursors if cursor is not None]) >= n:
            min_doc_id = min([cursor[0].document_id for cursor in all_cursors])
            ranker.reset(min_doc_id)
            # the non-exhausted posting lists that point to the lowest document ID
            frontier = [i for i, cursor in enumerate(all_cursors) if cursor is not None and cursor[0].document_id == min_doc_id]
            # check if the frontier satisfies the n of m criterion
            if len(frontier) >= n:
                cursors_to_be_ranked = [[cursor, term] for cursor, term in (all_cursors[i] for i in frontier)]
                for cursor in cursors_to_be_ranked:
                    posting = cursor[0]
                    term = cursor[1]
                    multiplicity = query_counter[term]
                    ranker.update(term, multiplicity, posting)

                sieve.sift(ranker.evaluate(), min_doc_id)

            # update the frontier
            for i in frontier:
                term = all_cursors[i][1]
                posting_list = None
                # ensures matching generator is advanced, since all_cursors becomes shorter than posting_lists
                for pl in posting_lists:
                    if pl[1] == term:
                        posting_list = pl[0]
                next_posting = next(posting_list, None)
                all_cursors[i][0] = next_posting

            # remove potential exhausted posting lists
            all_cursors = [cursor for cursor in all_cursors if cursor[0] is not None]

        top_results = sieve.winners()

        for score, doc_id in top_results:
            document = self.__corpus.get_document(doc_id)
            yield {'score': score, 'document': document}

