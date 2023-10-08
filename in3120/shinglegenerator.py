#!/usr/bin/python
# -*- coding: utf-8 -*-

from typing import Iterator, Tuple
from .tokenizer import Tokenizer
import re


class ShingleGenerator(Tokenizer):
    """
    Tokenizes a buffer into overlapping shingles having a specified width. For example, the
    3-shingles for "mouse" are {"mou", "ous", "use"}.

    If the buffer is shorter than the shingle width then this produces a single shorter-than-usual
    shingle.

    The current implementation is simplistic and not whitespace- or punctuation-aware,
    and doesn't treat the beginning or end of the buffer in a special way.
    """

    def __init__(self, width: int):
        assert width > 0
        self.__width = width

    def ranges(self, buffer: str) -> Iterator[Tuple[int, int]]:
        """
        Locates where the shingles begin and end.
        """
        for i in range(len(buffer)):
            if len(buffer) < self.__width:
                yield i, len(buffer)
                break
            yield i, i + self.__width
            if i >= len(buffer) - self.__width:
                break