#!/usr/bin/python
# -*- coding: utf-8 -*-

from typing import Iterator
from .posting import Posting


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

        # Start at the head.
        current1 = next(p1, None)
        current2 = next(p2, None)

        # We're doing an AND, so we can abort as soon as we exhaust one of
        # the posting lists.
        while current1 and current2:

            # Increment the smallest one. Yield if we have a match.
            if current1.document_id == current2.document_id:
                yield current1
                current1 = next(p1, None)
                current2 = next(p2, None)
            elif current1.document_id < current2.document_id:
                current1 = next(p1, None)
            else:
                current2 = next(p2, None)

    @staticmethod
    def union(p1: Iterator[Posting], p2: Iterator[Posting]) -> Iterator[Posting]:
        """
        A generator that yields a simple OR of two posting lists, given
        iterators over these.

        The posting lists are assumed sorted in increasing order according
        to the document identifiers.
        """

        # Start at the head.
        current1 = next(p1, None)
        current2 = next(p2, None)

        # We're doing an OR. First handle the case where neither posting
        # list is exhausted.
        while current1 and current2:

            # Yield the smallest one.
            if current1.document_id == current2.document_id:
                yield current1
                current1 = next(p1, None)
                current2 = next(p2, None)
            elif current1.document_id < current2.document_id:
                yield current1
                current1 = next(p1, None)
            else:
                yield current2
                current2 = next(p2, None)

        # At least one of the lists are exhausted. Yield the remaining tail(s), if any.
        if current1:
            yield current1
            yield from p1
        if current2:
            yield current2
            yield from p2
