from collections import Iterator
from itertools import chain
from typing import List, Dict, Iterable, Tuple

from numpy import nan
from pyitlib import discrete_random_variable as drv

from netzob.Model.Vocabulary.Messages.AbstractMessage import AbstractMessage
from netzob.Model.Vocabulary.Messages.L4NetworkMessage import L4NetworkMessage

from nemere.inference.segments import MessageAnalyzer

class NgramIterator(Iterator):
    """
    Iterate over the byte n-grams in message.

    FH, Section 3.1.2
    """

    def __init__(self, message: AbstractMessage, n=3):
        self._message = message
        self._n = n
        self.__offset = -1

    def __iter__(self):
        self.__offset = -1
        return self

    def __next__(self) -> bytes:
        self.__offset += 1
        if self.__offset > len(self._message.data) - self._n:
            raise StopIteration()
        return self._message.data[self.__offset:self.__offset+self._n]

    @property
    def offset(self):
        return self.__offset

    @property
    def exhausted(self):
        return self.__offset > len(self._message.data) - self._n


class Flows(object):
    """
    In FH, a flow is defined by the 5-tuple: Layer-4 Protocol, Source IP, Source Port, Destination IP, Destination IP
    """

    def __init__(self, messages: List[L4NetworkMessage]):
        self._messages = messages
        self._flows = self._identify()

    def _identify(self):
        """
        identify flows
        """
        flows = dict()  # type: Dict[Tuple, List[L4NetworkMessage]]
        # client is initiator, sort by packet date
        for msg in sorted(self._messages, key=lambda m: m.date):
            src = msg.source.rpartition(':')
            dst = msg.destination.rpartition(':')
            srcAddress = src[0]
            dstAddress = dst[0]
            srcPort = src[2]
            dstPort = dst[2]
            keytuple = (msg.l4Protocol, srcAddress, dstAddress, srcPort, dstPort)
            if keytuple not in flows:
                flows[keytuple] = list()
            flows[keytuple].append(msg)
        return flows

    def dialogs(self):
        """
        find pairs of flows with src/dst reversed to each other
        """
        dialogs = dict()
        for keytuple in self._flows.keys():
            # exchange src and dst addresses and ports
            rkeytuple = (keytuple[0], keytuple[2], keytuple[1], keytuple[4], keytuple[3])
            if rkeytuple in dialogs:
                if dialogs[rkeytuple] is not None:
                    raise Exception("Strange things happened here.")
                # identify the flow starting earlier as client (key in dialogs), the other as server (value in dialogs)
                if self._flows[rkeytuple][0].date < self._flows[keytuple][0].date:
                    dialogs[rkeytuple] = keytuple
                else:
                    del dialogs[rkeytuple]
                    dialogs[keytuple] = rkeytuple
            else:
                dialogs[keytuple] = None
        return dialogs

    def splitDirections(self):
        """
        Split list of messages into directions S2C and C2S based on flow information.
        Ignores all flows that have no reverse direction.

        FH, Section 2, Footnote 1
        """
        dialogs = self.dialogs()
        # merge all client flows into one and all server flows into another list of messages
        c2s = list(chain.from_iterable(self._flows[keytuple] for keytuple in dialogs.keys() if dialogs[keytuple] is not None))
        s2c = list(chain.from_iterable(self._flows[keytuple] for keytuple in dialogs.values() if keytuple is not None))
        return c2s, s2c

    def matchQueryRespone(self):
        """

        mqr = flows.matchQueryRespone()
        print(tabulate([(q.date, r.date) for q, r in mqr.items()]))

        """
        dialogs = self.dialogs()
        qr = dict()

        for keytuple in dialogs.keys():
            if dialogs[keytuple] is None:
                continue
            qlist = self._flows[keytuple].copy()
            rlist = self._flows[dialogs[keytuple]].copy()

            # assume qlist and rlist are sorted by query.date and resp.date
            prevquery = None
            for query in qlist:
                respFound = False
                for resp in rlist:
                    # first response later than query
                    if query.date < resp.date:
                        qr[query] = resp
                        respFound = True
                        break
                if not respFound:
                    continue
                # if the response to query seems to be the same than to the previous query...
                if prevquery is not None and qr[query] == qr[prevquery]:
                    # ... ignore the earlier query since a response message in between seems to have gone missing.
                    del qr[prevquery]
                prevquery = query
        return qr


def entropyVertical(messages: List[L4NetworkMessage], n=1):
    """
    Find offsets of n-grams (with the same offset in different messages of the list), that are not constant and not
    random, i. e., that have a entropy > 0 and < x (threshold?)

    FH, Section 3.2.1
    """
    ngIters = [NgramIterator(msg, n) for msg in messages]
    vEntropy = list()

    for ngrams in zip(*ngIters):
        vEntropy.append(MessageAnalyzer.calcEntropy(ngrams, 256))

    return vEntropy


def intsFromNgrams(ngrams: List[bytes], endianness='big'):
    return [int.from_bytes(b, endianness) for b in ngrams]


def pyitEntropyVertical(messages: List[L4NetworkMessage], n=1, endianness='big'):
    """
    Find offsets of n-grams (with the same offset in different messages of the list), that are not constant and not
    random, i. e., that have a entropy > 0 and < x (threshold?)

    >>> entropyVertical(messages) == pyitEntropyVertical(messages)

    FH, Section 3.2.1
    """
    ngIters = [NgramIterator(msg, n) for msg in messages]
    vEntropy = list()

    for ngrams in zip(*ngIters):
        # int.from_bytes is necessary because of numpy's issue with null-bytes: #3878
        #   (https://github.com/numpy/numpy/issues/3878)
        vEntropy.append(drv.entropy(intsFromNgrams(ngrams, endianness))/(n*8))

    return vEntropy


def mutualInformation(qInts: List[List[int]], rInts: List[List[int]]):
    """

    :param qInts: List of n-grams as int-list
    :param rInts: List of n-grams as int-list
    :return:
    """
    qEntropy = drv.entropy(qInts)
    if qEntropy != 0:
        return drv.information_mutual(qInts, rInts) / qEntropy
    else:
        return nan


def qrAssociationCorrelation(mqr: Dict[L4NetworkMessage, L4NetworkMessage], n=1):
    """
    Take the matched query-response pairs (mqr) and associate ngram offsets by mutual information as correlation
    metric.

    # TODO optimize efficiency by supporting a input filter, i. e.,
        calculate mutual information only for given ngram offsets

    :param mqr: Matched query-response pairs
    :param n: The length of the n-grams to use (in bytes)
    :returns: Offset => causality value
    """
    mutInf = dict()
    qIterators, rIterators = list(), list()
    for qrPair in mqr.items():
        qIterators.append(NgramIterator(qrPair[0], n))
        rIterators.append(NgramIterator(qrPair[1], n))
    while not all(qiter.exhausted for qiter in qIterators) or all(riter.exhausted for riter in rIterators):
        qNgrams = list()
        rNgrams = list()
        # get two lists of ngrams with the same offset, one for queries, one for responses
        for qIter, rIter in zip(qIterators, rIterators):
            try:
                qNgram = next(qIter)
                rNgram = next(rIter)
            except StopIteration:
                # there are no more ngrams for query or response for this pair of Q/R messages
                continue
            qNgrams.append(qNgram)
            rNgrams.append(rNgram)
            # print("Q offset:", qIter.offset)  # should be the same for all iterators in one while loop
            # print("R offset:", rIter.offset, "\n")
        if len(qNgrams) == 0 or len(rNgrams) == 0:
            break
        # print(qNgrams)
        # print(rNgrams, "\n")
        qInts = intsFromNgrams(qNgrams)
        rInts = intsFromNgrams(rNgrams)
        mutInf[qIter.offset] = mutualInformation(qInts, rInts)
    return mutInf


def verticalByteMerge(mqr: Dict[L4NetworkMessage, L4NetworkMessage], offsets: Iterable[int]):
    """
    Returns two lists of integer-list representations of byte strings, one from all queries and one from all responses,
    containing the bytes at all offsets given as parameter.

    :param mqr:
    :param offsets:
    :return:
    """
    from itertools import compress

    sortedOffs = sorted(offsets)
    qMerge = list()
    rMerge = list()
    for query, resp in mqr.items():
        # int.from_bytes is necessary because of numpy's issue with null-bytes: #3878
        #   (https://github.com/numpy/numpy/issues/3878)
        qMerge.append(int.from_bytes(bytes(query.data[o] for o in sortedOffs), 'big'))
        rMerge.append(int.from_bytes(bytes(resp.data[o] for o in sortedOffs), 'big'))

    return qMerge, rMerge


def list2ranges(offsets: List[int]):
    """
    Generate ranges from a list of integer values. The ranges denote the starts end ends of any subsequence of
    adjacent values, e. g. the list [1,2,3,6,7,20] would result in the ranges [(1,3),(6,7),(20,20)]

    :param offsets:
    :return:
    """
    soffs = sorted(offsets)
    ranges = list()  # type: List[Tuple[int,int]]
    if len(soffs) == 0:
        return ranges
    else:
        assert soffs[0] >= 0, "Offsets must be positive numbers."
    if len(soffs) <= 1:
        return [(soffs[0],soffs[0])]
    start = soffs[0]
    last = soffs[0]
    for offs in soffs[1:]:
        if offs > last + 1:
            ranges.append((start,last))
            # start a new range
            start = offs
        last = offs
    ranges.append((start, last))

    return ranges