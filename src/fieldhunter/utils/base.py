from collections import Iterator
from itertools import chain
from typing import List, Dict, Iterable, Tuple, Union

import IPython
from numpy import nan
from pyitlib import discrete_random_variable as drv

from netzob.Model.Vocabulary.Messages.AbstractMessage import AbstractMessage
from netzob.Model.Vocabulary.Messages.L4NetworkMessage import L4NetworkMessage

from nemere.inference.segments import MessageAnalyzer



class NgramIterator(Iterator):
    """
    Iterate over the byte n-grams in message.

    FH, Section 3.1.2


    >>> from fieldhunter.utils.base import NgramIterator
    >>> from netzob.Model.Vocabulary.Messages.L4NetworkMessage import L4NetworkMessage
    >>> ngi = NgramIterator(L4NetworkMessage(b"QQQ456789"))
    >>> for ngram in ngi:
    ...     print(ngram, ngi.offset, ngi.exhausted, ngi.lookahead)
    b'QQQ' 0 False True
    b'QQ4' 1 False True
    b'Q45' 2 False True
    b'456' 3 False True
    b'567' 4 False True
    b'678' 5 False True
    b'789' 6 False False
    >>> print(ngi.exhausted, ngi.lookahead)
    True False
    """

    def __init__(self, message: AbstractMessage, n=3):
        """

        :param message: The message of which to iterate the n-grams.
        :param n: The n in n-gram (length of chunk in bytes).
        """
        if not isinstance(message, AbstractMessage):
            raise ValueError("Parameter needs to be a Netzob message object (AbstractMessage).")
        self._message = message
        self._n = n
        self.__offset = -1

    __step = 1

    def __iter__(self):
        self.__offset = -1
        return self

    def __next__(self) -> bytes:
        self.__offset += NgramIterator.__step
        if self.exhausted:
            raise StopIteration()
        return self._message.data[self.__offset:self.__offset+self._n]

    @property
    def offset(self):
        """
        NgramIterator enumerates the offset of the n-gram its current iteration is taken from.

        :return: offset of the n-gram in the current iteration.
        """
        return self.__offset

    @property
    def exhausted(self):
        """
        :return: Indicates that the last iteration has occurred.
        """
        return self.__offset > len(self._message.data) - self._n

    @property
    def lookahead(self):
        """
        :return: True indicates that at least one more iteration is contained in this iterator.
        """
        return self.__offset + NgramIterator.__step <= len(self._message.data) - self._n


class Flows(object):
    # noinspection PyUnresolvedReferences
    """
    In FH, a flow is defined by the 5-tuple: Layer-4 Protocol, Source IP, Destination IP, Source Port, Destination Port
    (FH, Section2, Footnote 1)

    >>> from tabulate import tabulate
    >>> from netzob.Model.Vocabulary.Messages.L4NetworkMessage import L4NetworkMessage
    >>> from fieldhunter.utils.base import Flows
    >>> messages = [
    ...    L4NetworkMessage(b"QQQ456789", l4Protocol="dummy", date=1445405280.01, l3SourceAddress="192.168.23.100", l3DestinationAddress="192.168.23.245", l4SourceAddress=10815, l4DestinationAddress=42),
    ...    L4NetworkMessage(b"RRR567890", l4Protocol="dummy", date=1445405280.03, l3SourceAddress="192.168.23.245", l3DestinationAddress="192.168.23.100", l4SourceAddress=42, l4DestinationAddress=10815),
    ...    L4NetworkMessage(b"QQQ7890AB", l4Protocol="dummy", date=1445405280.07, l3SourceAddress="192.168.23.100", l3DestinationAddress="192.168.23.245", l4SourceAddress=10815, l4DestinationAddress=42),
    ...    L4NetworkMessage(b"RRR567890", l4Protocol="dummy", date=1445405280.05, l3SourceAddress="192.168.23.245", l3DestinationAddress="192.168.23.100", l4SourceAddress=42, l4DestinationAddress=10815),
    ...    L4NetworkMessage(b"QQQ123456789", l4Protocol="dummy", date=1445405280.11, l3SourceAddress="192.168.23.100", l3DestinationAddress="192.168.23.245", l4SourceAddress=1717, l4DestinationAddress=2323),
    ...    L4NetworkMessage(b"RRR890A", l4Protocol="dummy", date=1445405280.13, l3SourceAddress="192.168.23.1", l3DestinationAddress="192.168.23.100", l4SourceAddress=2323, l4DestinationAddress=1717),
    ...    L4NetworkMessage(b"QQQ6789", l4Protocol="dummy", date=1445405280.17, l3SourceAddress="192.168.23.1", l3DestinationAddress="192.168.23.245", l4SourceAddress=1717, l4DestinationAddress=2323),
    ...    L4NetworkMessage(b"RRR890ABCDEFGH", l4Protocol="dummy", date=1445405280.23, l3SourceAddress="192.168.23.245", l3DestinationAddress="192.168.23.100", l4SourceAddress=2323, l4DestinationAddress=1717)
    ... ]
    >>> # for the sake of the test case, the messages RRR890A and QQQ6789 have src IPs that rules them out as
    >>> #   valid conversations, they should not be contained in the conversion lists below.
    >>> flows = Flows(messages)
    >>> mqr = flows.matchQueryResponse()
    >>> print(tabulate([ (q.date, r.date) for q, r in mqr.items() ], floatfmt=""))
    -------------  -------------
    1445405280.01  1445405280.03
    1445405280.11  1445405280.23
    -------------  -------------
    >>> print(tabulate( [(list(flowtuple) + [b" ".join(msg.data for msg in msglst)]) for flowtuple, msglst in flows.conversations().items()] ))
    -----  --------------  --------------  -----  ----  ---------------------------------------
    dummy  192.168.23.100  192.168.23.245  10815    42  QQQ456789 QQQ7890AB RRR567890 RRR567890
    dummy  192.168.23.100  192.168.23.245   1717  2323  QQQ123456789 RRR890ABCDEFGH
    -----  --------------  --------------  -----  ----  ---------------------------------------
    >>> print(tabulate( [(list(flowtuple) + [b" ".join(msg.data for msg in msglst)]) for flowtuple, msglst in flows.c2sInConversations().items()] ))
    -----  --------------  --------------  -----  ----  -------------------
    dummy  192.168.23.100  192.168.23.245  10815    42  QQQ456789 QQQ7890AB
    dummy  192.168.23.100  192.168.23.245   1717  2323  QQQ123456789
    -----  --------------  --------------  -----  ----  -------------------
    >>> print(tabulate( [(list(flowtuple) + [b" ".join(msg.data for msg in msglst)]) for flowtuple, msglst in flows.s2cInConversations().items()] ))
    -----  --------------  --------------  -----  ----  -------------------
    dummy  192.168.23.100  192.168.23.245  10815    42  RRR567890 RRR567890
    dummy  192.168.23.100  192.168.23.245   1717  2323  RRR890ABCDEFGH
    -----  --------------  --------------  -----  ----  -------------------
    """

    def __init__(self, messages: List[L4NetworkMessage]):
        self._messages = messages
        self._flows = self._identify()

    @property
    def messages(self):
        return self._messages

    def _identify(self) -> Dict[Tuple, List[L4NetworkMessage]]:
        """
        Identify flows.

        :return A dict mapping the 5-tuple
            (Layer-4 Protocol, Source IP, Destination IP, Source Port, Destination Port)
            to the list of addresses in the flow denoted by the 5-tuple.
        """
        flows = dict()  # type: Dict[Tuple[str,str,str,str,str], List[L4NetworkMessage]]
        # client is initiator, sort by packet date
        for msg in sorted(self._messages, key=lambda m: m.date):  # type: L4NetworkMessage
            if not isinstance(msg, L4NetworkMessage):
                raise TypeError("To identify flows, all messages need to be from a known encapsulation with known "
                                "network and transport layer protocols. No flow determined for "
                                f"{type(msg).__name__}:\n{msg}")
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

    @property
    def flows(self):
        return self._flows

    def conversations(self) -> Dict[Tuple, List[AbstractMessage]]:
        """
        "A conversation is formed of the two flows in opposite direction..." (FH, Section2, Footnote 1)
        :return: Dict of conversations with the c2s flow tuple as key.
        """
        return {qkey: self._flows[qkey] + self._flows[rkey]
                for qkey,rkey in self._dialogs().items() if rkey is not None}

    def c2sInConversations(self) -> Dict[Tuple, List[AbstractMessage]]:
        """
        "A conversation is formed of the two flows in opposite direction..." (FH, Section2, Footnote 1)
        :return: Dict of c2s messages per conversation with the c2s flow tuple as key.
        """
        return {qkey: self._flows[qkey] for qkey,rkey in self._dialogs().items() if rkey is not None}

    def s2cInConversations(self) -> Dict[Tuple, List[AbstractMessage]]:
        """
        "A conversation is formed of the two flows in opposite direction..." (FH, Section2, Footnote 1)
        :return: Dict of s2c messages per conversation with the c2s flow tuple as key.
        """
        return {qkey: self._flows[rkey] for qkey,rkey in self._dialogs().items() if rkey is not None}

    def _dialogs(self) -> Dict[Tuple,Union[Tuple,None]]:
        """
        find pairs of flows with src/dst and reversed to each other.
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

    def splitDirections(self) -> Tuple[List[AbstractMessage],List[AbstractMessage]]:
        """
        Split list of messages into directions S2C and C2S based on flow information.
        Ignores all flows that have no reverse direction.

        FH, Section 2, Footnote 1

        :return Lists of messages, the first is client-to-server, the second is server-to-client
        """
        # merge all client flows into one and all server flows into another list of messages
        c2s = list(chain.from_iterable(self.c2sInConversations().values()))
        s2c = list(chain.from_iterable(self.s2cInConversations().values()))
        return c2s, s2c

    def matchQueryResponse(self):
        """
        Match queries with responses in the flows by identifying
        for each client-to-server message (query) the server-to-client message (response)
        that has the closest subsequent transmission time.
        """
        dialogs = self._dialogs()
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


def ngramEntropy(messages: List[AbstractMessage], n=1):
    """
    The vertical entropies for each offset of all the n-grams at the same offset throughout all messages.
    Own entropy calculation implementation. See #pyitEntropyVertical

    FH, Section 3.2.1
    """
    ngIters = [NgramIterator(msg, n) for msg in messages]
    vEntropy = list()

    for ngrams in zip(*ngIters):
        vEntropy.append(MessageAnalyzer.calcEntropy(ngrams, 256))

    return vEntropy


def intsFromNgrams(ngrams: Iterable[bytes], endianness='big') -> List[int]:
    r"""
    Convert an iterable of byte n-grams into a single integer per n-gram.
    This is useful to simplify working aroung numpy's issue with null-bytes:
        Issue #3878 (https://github.com/numpy/numpy/issues/3878)

    >>> from fieldhunter.utils.base import intsFromNgrams
    >>> ngramlist = [b"\x00\x00\x00", b"\x00\x11\x00", b"\xab\x00\x00", b"\xab\x11\x23", b"\x08\x15"]
    >>> ifn = intsFromNgrams(ngramlist)
    >>> # noinspection PyUnresolvedReferences
    >>> [hex(val) for val in ifn]
    ['0x0', '0x1100', '0xab0000', '0xab1123', '0x815']

    :param ngrams: Iterable of n-grams, one bytes string per n-gram
    :param endianness: The endianness to use to interpret the bytes.
    :return: List of integers, one for with the value of each bytes string n-gram.
    """
    return [int.from_bytes(b, endianness) for b in ngrams]


def pyitNgramEntropy(messages: List[AbstractMessage], n=1, endianness='big'):
    """
    The vertical entropies for each offset of all the n-grams at the same offset throughout all messages.
    Implementation of entropy calculation from pyitlib. See #entropyVertical

    FH, Section 3.2.1

    >>> from netzob.Model.Vocabulary.Messages.L4NetworkMessage import L4NetworkMessage
    >>> from fieldhunter.utils.base import pyitNgramEntropy, ngramEntropy
    >>> messageList = [
    ...    L4NetworkMessage(b"QQQ456789"), L4NetworkMessage(b"RRR567890"), L4NetworkMessage(b"QQQ7890AB"),
    ...    L4NetworkMessage(b"RRR567890"), L4NetworkMessage(b"QQQ123456789"), L4NetworkMessage(b"RRR890A"),
    ...    L4NetworkMessage(b"QQQ6789"), L4NetworkMessage(b"RRR890ABCDEFGH")
    ... ]
    >>> ngramEntropy(messageList) == pyitNgramEntropy(messageList)
    True

    :param messages: The list of messages to get the n-grams from.
    :param n: The n in n-gram.
    :param endianness: Endianness to interpret the n-grams in.
    """
    ngIters = [NgramIterator(msg, n) for msg in messages]
    vEntropy = list()

    for ngrams in zip(*ngIters):  # type: List[bytes]
        # int.from_bytes is necessary because of numpy's issue with null-bytes: #3878
        #   (https://github.com/numpy/numpy/issues/3878)
        vEntropy.append(drv.entropy(intsFromNgrams(ngrams, endianness))/(n*8))

    return vEntropy


def mutualInformationNormalized(qInts: Union[List[List[int]],List[int]], rInts: Union[List[List[int]],List[int]]):
    """
    Calculate the Mutual Information between two lists of n-grams, e.g.,
        one list being queries and the other the according responses, normalized to the queries' entropy.
    Mutual information measures the information shared between the two lists. (FH, Section 3.2.1)

    >>> from fieldhunter.utils.base import mutualInformationNormalized, intsFromNgrams
    >>> queryNgramsConst = [b'QQQ', b'QQQ', b'QQQ', b'QQQ']
    >>> respoNgramsConst = [b'RRR', b'RRR', b'RRR', b'RRR']
    >>> queryNgramsCorr = [b'42', b'40', b'42', b'23', b'17']
    >>> respoNgramsCorr = [b'24', b'04', b'24', b'32', b'71']
    >>> queryNgramsPart = [b'42', b'40', b'42', b'23', b'17']
    >>> respoNgramsPart = [b'04', b'04', b'04', b'32', b'71']
    >>> # query and response n-grams are always constant: This allows no conclusion about any correlation => nan
    >>> mutualInformationNormalized(intsFromNgrams(queryNgramsConst), intsFromNgrams(respoNgramsConst))
    nan
    >>> # query and response n-grams always have corresponding values => perfectly correlated
    >>> mutualInformationNormalized(intsFromNgrams(queryNgramsCorr), intsFromNgrams(respoNgramsCorr))
    1.0
    >>> # some query and response n-grams have corresponding values, multiple other queries have the same responses.
    >>> mutualInformationNormalized(intsFromNgrams(queryNgramsPart), intsFromNgrams(respoNgramsPart))
    0.7133...

    :param qInts: List of (n-grams as int-list) or one int per realization
    :param rInts: List of (n-grams as int-list) or one int per realization
    :return:
    """
    assert len(qInts) > 0, "Entropy requires at least one query realization"
    assert len(rInts) > 0, "Entropy requires at least one reply realization"
    assert len(qInts) == len(rInts), "Mutual information requires the same amount of query and reply realizations"
    qEntropy = drv.entropy(qInts)
    if qEntropy != 0:
        return drv.information_mutual(qInts, rInts) / qEntropy
    else:
        return nan


def qrAssociationCorrelation(mqr: Dict[L4NetworkMessage, L4NetworkMessage], n=1):
    """
    Take the matched query-response pairs (mqr)
    and associate n-gram offsets by mutual information as correlation metric.

    >>> from tabulate import tabulate
    >>> from netzob.Model.Vocabulary.Messages.L4NetworkMessage import L4NetworkMessage
    >>> from fieldhunter.utils.base import qrAssociationCorrelation
    >>> matchedqr = {
    ...    L4NetworkMessage(b"QQQ456789"): L4NetworkMessage(b"RRR567890"),
    ...    L4NetworkMessage(b"QQQ7890AB"): L4NetworkMessage(b"RRR567890"),
    ...    L4NetworkMessage(b"QQQ123456789"): L4NetworkMessage(b"RRR890A"),
    ...    L4NetworkMessage(b"QQQ6789"): L4NetworkMessage(b"RRR890ABCDEFGH"),
    ... }
    >>> qrAC = qrAssociationCorrelation(matchedqr)
    >>> print(tabulate(qrAC.items()))
    -  -----
    0  nan
    1  nan
    2  nan
    3    0.5
    4    0.5
    5    0.5
    6    0
    -  -----

    For the first 3 bytes the normalized mutual information is undefined: see #mutualInformationNormalized()

    # TODO optimize efficiency by supporting an input filter, i. e.,
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
        # get two lists of n-grams with the same offset, one for queries, one for responses
        for qIter, rIter in zip(qIterators, rIterators):
            if not qIter.lookahead or not rIter.lookahead:
                # there are no more n-grams for either query or response for this pair of Q/R messages
                continue
            # fetch the next iteration for both messages in parallel.
            # A StopIteration should never occur here, since we check if the iterators are soon being exhausted before.
            qNgrams.append(next(qIter))
            rNgrams.append(next(rIter))
            # print("Q offset:", qIter.offset)  # should be the same for all iterators in one while loop
            # print("R offset:", rIter.offset, "\n")
        if len(qNgrams) == 0 or len(rNgrams) == 0:
            break
        # print(qNgrams)
        # print(rNgrams, "\n")
        qInts = intsFromNgrams(qNgrams)
        rInts = intsFromNgrams(rNgrams)
        # qIter and rIter are always set here, otherwise the break on (len(qNgrams) == 0 or len(rNgrams) == 0)
        #   above would have been triggered
        # noinspection PyUnboundLocalVariable
        if qIter.offset != rIter.offset:
            # NgramIterator remembers the offset of its current iteration. This must be the same in both messages.
            raise RuntimeError("The offsets in qrAssociationCorrelation calculation do not match:"
                               f"{qIter.offset} {rIter.offset}\n{qNgrams}\n{rNgrams}")
        mutInf[qIter.offset] = mutualInformationNormalized(qInts, rInts)
    return mutInf


def verticalByteMerge(mqr: Dict[L4NetworkMessage, L4NetworkMessage], offsets: Iterable[int]):
    # noinspection PyUnresolvedReferences
    """
    Returns two lists of integer-list representations of byte strings,
    one from all queries and one from all responses,
    containing the bytes at all offsets given as parameter.

    >>> from fieldhunter.utils.base import verticalByteMerge
    >>> messageMap = {
    ...    L4NetworkMessage(b"QQQ456789"): L4NetworkMessage(b"RRR567890"),
    ...    L4NetworkMessage(b"QQQ7890AB"): L4NetworkMessage(b"RRR567890"),
    ...    L4NetworkMessage(b"QQQ123456789"): L4NetworkMessage(b"RRR890A"),
    ...    L4NetworkMessage(b"QQQ6789"): L4NetworkMessage(b"RRR890ABCDEFGH"),
    ... }
    >>> verticalByteMerge(messageMap, [1])
    ([81, 81, 81, 81], [82, 82, 82, 82])
    >>> verticalByteMerge(messageMap, [1,2,3])
    ([5329204, 5329207, 5329201, 5329206], [5394997, 5394997, 5395000, 5395000])
    >>> qMsgs, rMsgs = verticalByteMerge(messageMap, [3,5,6])
    >>> # ints converted back to the bytes number 3, 5, and 6 from the keys in messageMap
    >>> [int.to_bytes(val, 3, 'big') for val in qMsgs]
    [b'467', b'790', b'134', b'689']
    >>> # ints converted back to the bytes number 3, 5, and 6 from the values in messageMap
    >>> [int.to_bytes(val, 3, 'big') for val in rMsgs]
    [b'578', b'578', b'80A', b'80A']

    :param mqr: Dict that maps one message to another.
    :param offsets: List of offsets for which the byte values should be returned.
        The offset must exist in all messages.
    :return: Two lists of integer representations of the byte values at the given offsets,
        one list for the keys and one for the values of the input dict.
    :raises IndexError: If an offset does not exist in any message.
    """
    sortedOffs = sorted(offsets)
    qMerge = list()
    rMerge = list()
    for query, resp in mqr.items():
        # int.from_bytes is necessary because of numpy's issue with null-bytes: #3878
        #   (https://github.com/numpy/numpy/issues/3878)
        qMerge.append(int.from_bytes(bytes(query.data[o] for o in sortedOffs), 'big'))
        rMerge.append(int.from_bytes(bytes(resp.data[o] for o in sortedOffs), 'big'))
    return qMerge, rMerge


def iterateSelected(toIter: Iterator, selectors: List[int]):
    """
    Only return selected iterations from an iterator.

    >>> from fieldhunter.utils.base import iterateSelected
    >>> bytesTuple = iter((b'QQQ456789', b'RRR567890', b'QQQ7890AB', b'RRR567890', b'QQQ123456789', b'RRR890A'))
    >>> bt2357 = iterateSelected(bytesTuple, [2,3,5,7])
    >>> next(bt2357)
    b'QQQ7890AB'
    >>> next(bt2357)
    b'RRR567890'
    >>> next(bt2357)
    b'RRR890A'
    >>> next(bt2357)  # doctest: +IGNORE_EXCEPTION_DETAIL
    Traceback (most recent call last):
    StopIteration

    :param toIter: The iterator to traverse.
    :param selectors: The list of iteration indices to return.
    :return: A generator for all iterations in toIter that have an "index" selected by selectors.
    """
    return (element for offset, element in enumerate(toIter) if offset in selectors)


def list2ranges(offsets: List[int]):
    """
    Generate ranges from a list of integer values. The ranges denote the starts and lengths of any subsequence of
    adjacent values, e. g. the list [1,2,3,6,7,20] would result in the ranges [(1,3),(6,2),(20,1)]

    >>> from fieldhunter.utils.base import list2ranges
    >>> list2ranges([1,2,3,6,7,20])
    [(1, 3), (6, 2), (20, 1)]
    >>> list2ranges([2,3,6,11,12,13,23,24,25,26])
    [(2, 2), (6, 1), (11, 3), (23, 4)]
    >>> list2ranges([2])
    [(2, 1)]
    >>> list2ranges([])
    []
    >>> list2ranges([-2])  # doctest: +IGNORE_EXCEPTION_DETAIL
    Traceback (most recent call last):
    ValueError: Offsets must be positive numbers.

    :param offsets: list of integers
    :return: list of ranges (tuples of offset and length) of consecutive the offsets.
    """
    soffs = sorted(offsets)
    ranges = list()  # type: List[Tuple[int,int]]
    # offsets empty
    if len(soffs) == 0:
        return ranges
    if soffs[0] < 0:
        raise ValueError("Offsets must be positive numbers.")
    # only one offset
    if len(soffs) == 1:
        return [(soffs[0],1)]
    start = soffs[0]
    last = soffs[0]
    for offs in soffs[1:]:
        if offs > last + 1:
            ranges.append((start, last - start + 1))
            # start a new range
            start = offs
        last = offs
    # append dangling start/last
    ranges.append((start, last - start + 1))

    return ranges


def ngramIsOverlapping(o0, n0, o1, n1):
    """
    Check if two ranges are overlapping. The ranges are defined by offset and length each.

    >>> ngramIsOverlapping(2,2,0,3)
    True
    >>> ngramIsOverlapping(2,2,0,2)
    False
    >>> ngramIsOverlapping(2,2,3,2)
    True
    >>> ngramIsOverlapping(2,2,4,2)
    False

    :param o0: Offset of n-gram 0
    :param n0: Length (n) of n-gram 0
    :param o1: Offset of n-gram 1
    :param n1: Length (n) of n-gram 1
    :return: True if overlapping, false otherwise
    """
    return o1 + n1 - 1 >= o0 and o1 < o0 + n0

