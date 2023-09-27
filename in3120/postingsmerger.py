#!/usr/bin/python
# -*- coding: utf-8 -*-

from typing import Iterator
from .posting import Posting
import heapq

class PostingsMerger:
    """
    Utility class for merging posting lists.

    It is currently left unspecified what to do with the term frequency field
    in the returned postings when document identifiers overlap. Different
    approaches are possible, e.g., an arbitrary one of the two postings could
    be returned, or the posting having the smallest/largest term frequency, or
    a new one that produces an averaged value, or something else.
    """

    @staticmethod
    def intersection(p1: Iterator[Posting], p2: Iterator[Posting]) -> Iterator[Posting]:
        """
        A generator that yields a simple AND of two posting lists, given
        iterators over these.

        The posting lists are assumed sorted in increasing order according
        to the document identifiers.
        """
        posting_1 = next(p1, None)
        posting_2 = next(p2, None)
        while posting_1 and posting_2:
            if posting_1.document_id == posting_2.document_id:
                yield posting_1
                posting_1 = next(p1, None)
                posting_2 = next(p2, None)
            elif posting_1.document_id < posting_2.document_id:
                posting_1 = next(p1, None)
            else:
                posting_2 = next(p2, None)

    @staticmethod
    def set_difference(p1: Iterator[Posting], p2: Iterator[Posting]) -> Iterator[Posting]:
        """
        A generator that yields an AND NOT of two posting lists, given iterators over these.
        The posting lists are assumed sorted in increasing order according to the document identifiers.
        """
        posting_1 = next(p1, None)
        posting_2 = next(p2, None)
        while posting_1 and posting_2:
            if posting_1.document_id < posting_2.document_id:
                yield posting_1
                posting_1 = next(p1, None)
            elif posting_1.document_id > posting_2.document_id:
                posting_2 = next(p2, None)
            else:
                posting_1 = next(p1, None)
                posting_2 = next(p2, None)
        while posting_1:
            yield posting_1
            posting_1 = next(p1, None)

    @staticmethod
    def union(p1: Iterator[Posting], p2: Iterator[Posting]) -> Iterator[Posting]:
        """
        A generator that yields a simple OR of two posting lists, givenS
        iterators over these.

        The posting lists are assumed sorted in increasing order according
        to the document identifiers.
        """
        posting_1 = next(p1, None)
        posting_2 = next(p2, None)
        prev_posting = None
        while posting_1 or posting_2:
            # if posting_2 is the smallest or only posting
            if posting_1 is None or (posting_2 and posting_2.document_id < posting_1.document_id):
                # check if the posting was yielded in the previous iteration
                if posting_2 != prev_posting:
                    yield posting_2
                    prev_posting = posting_2
                # advance posting_2
                posting_2 = next(p2, None)
            # if posting_1 is the smallest or only posting
            elif posting_2 is None or posting_1.document_id < posting_2.document_id:
                # check if the posting was yielded in the previous iteration
                if posting_1 != prev_posting:
                    yield posting_1
                    prev_posting = posting_1
                # advance posting_1
                posting_1 = next(p1, None)
            # if the postings' docIDs are the same
            else:
                # check if posting_1 was yielded in the previous iteration
                if posting_1 != prev_posting:
                    yield posting_1
                    prev_posting = posting_1
                # advance both postings
                posting_1 = next(p1, None)
                posting_2 = next(p2, None)
