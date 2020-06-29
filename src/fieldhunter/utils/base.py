from collections import Iterator
from itertools import chain
from typing import List, Dict

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


def splitDirections(messages: List[L4NetworkMessage]):
    """
    Split list of messages into directions S2C and C2S based on flow information.
    In FH, a flow is defined by the 5-tuple: Layer-4 Protocol, Source IP, Source Port, Destination IP, Destination IP
    We reduce this to Layer-4 Protocol, Source IP, Destination IP (assuming, the traces are filtered for one protocol
    already) since, e. g., DNS clients use random ports.

    Ignores all flows that have no reverse direction.

    FH, Section 2, Footnote 1
    """
    # identify flows
    flows = dict()  # type: Dict[Tuple, List[L4NetworkMessage]]
    # client is initiator, sort by packet date
    for msg in sorted(messages, key=lambda m: m.date):
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

    # find pairs of flows with src/dst reversed to each other
    dialogs = dict()
    for keytuple in flows.keys():
        # exchange src and dst addresses and ports
        rkeytuple = (keytuple[0], keytuple[2], keytuple[1], keytuple[4], keytuple[3])
        if rkeytuple in dialogs:
            if dialogs[rkeytuple] is not None:
                raise Exception("Strange things happened here.")
            # identify the flow starting earlier as client (key in dialogs), the other as server (value in dialogs)
            if flows[rkeytuple][0].date > flows[keytuple][0].date:
                dialogs[rkeytuple] = keytuple
            else:
                del dialogs[rkeytuple]
                dialogs[keytuple] = rkeytuple
        else:
            dialogs[keytuple] = None

    # import IPython
    # # globals().update(locals())
    # IPython.embed()

    # merge all client flows into one and all server flows into another list of messages
    c2s = list(chain.from_iterable(flows[keytuple] for keytuple in dialogs.keys() if dialogs[keytuple] is not None))
    s2c = list(chain.from_iterable(flows[keytuple] for keytuple in dialogs.values() if keytuple is not None))

    return c2s, s2c


def entropyFilterVertical(messages: List[L4NetworkMessage], n=1):
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

