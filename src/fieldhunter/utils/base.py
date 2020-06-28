from collections import Iterator

from netzob.Model.Vocabulary.Messages.AbstractMessage import AbstractMessage


class NgramIterator(Iterator):


    def __init__(self, message: AbstractMessage, n=3):
        self._message = message
        self._n = n
        self.__offset = 0


    def __next__(self) -> bytes:
        if self.__offset > len(self._message.data) - self._n:
            raise StopIteration()
        self.__offset += 1
        return self._message.data[self.__offset-1:self.__offset+self._n-1]

