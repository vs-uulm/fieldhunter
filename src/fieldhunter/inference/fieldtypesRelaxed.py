"""
Infer message field types according to the FieldHunter paper Section 3.2
but with some relaxed thresholds and assumptions.

TODO introduce doctests to check critical functions in inference.fieldtypes
"""
from typing import List, Tuple, Dict, Iterable, ItemsView, Union
import random, logging
from itertools import groupby, product, chain, combinations
from collections import Counter
from abc import ABC, abstractmethod

import numpy
from scipy.stats import pearsonr
from pyitlib import discrete_random_variable as drv
from netzob.Model.Vocabulary.Messages.AbstractMessage import AbstractMessage
from netzob.Model.Vocabulary.Messages.L2NetworkMessage import L2NetworkMessage
from netzob.Model.Vocabulary.Messages.L4NetworkMessage import L4NetworkMessage

from nemere.inference.segments import TypedSegment

from fieldhunter.utils.base import qrAssociationCorrelation, verticalByteMerge, mutualInformationNormalized, \
    list2ranges, Flows, NgramIterator, iterateSelected, intsFromNgrams, \
    ngramIsOverlapping, pyitNgramEntropy
from fieldhunter.inference.fieldtypes import NonConstantNonRandomEntropyFieldType, FieldType, Accumulator
import fieldhunter.inference.fieldtypes as fieldtypes


logging.getLogger(__name__).setLevel(logging.DEBUG)


class MSGtype(fieldtypes.MSGtype):
    """
    Relaxed version of message type (MSG-Type) inference (FH, Section 3.2.1, Fig. 3 left).

    see .fieldtypes.MSGtype
    """
    causalityThresh = 0.7
    """
    FH, Sec. 3.2.1 says 0.8, but that leaves no candidates for our NTP traces
    Reduces TP and FP for SMB 100.
    """


class MSGlen(fieldtypes.MSGlen):
    """
    Relaxed version of message length (MSG-Len) inference (FH, Section 3.2.2, Fig. 3 center).

    see .fieldtypes.MSGlen
    """
    # coefficient threshold 0.6 (FH, Section 3.2.2)
    minCorrelation = 0.6
    # MSG-Len hypothesis threshold 0.9 (FH, Section 3.2.2)
    lenhypoThresh = 0.9

    def __init__(self, flows: Flows):
        super(NonConstantNonRandomEntropyFieldType, self).__init__()

        # The FH paper per direction wants to handle each direction separately, which is pointless for MSG-Len,
        # so we place all messages in one direction object.
        self._msgDirection = [type(self).Direction(flows.messages)]
        # TODO It might rather be useful to separate message types (distinct formats) in this manner.
        #   However, this requires combination with some message type classification approach. => Future Work.

    class Direction(fieldtypes.MSGlen.Direction):

        @staticmethod
        def _candidateIsAcceptable(solutionAcceptable: Dict[Tuple[AbstractMessage, AbstractMessage], bool],
                                   Xarray: numpy.ndarray):
            """
            FH does not require that either the values in X[0]/a or X[1]/b are equal for all X in Xes.
            Thus, different values are accepted, although a message length typically is calculated using the same
            multiplicator X[0], even if the offset X[1] may change, so X[0] must be a scalar value.

            Otherwise we end up with lots of FPs. Examples:
                * In SMB, the 'Msg. Len. Model Parameters' (a,b) == [1., 4.]
                  of the 4-gram at offset 0, 4 is nbss.length, i. e., a TP!
                  Offsets 16 and 22 are FP, but with diverging A and B vectors.
                * Another example: In DNS, the beginning of the queried name is a FP
                  (probably due to DNS' subdomain numbered separator scheme).

            Thus, we require that X[0] is the same constant value throughout the majority of checked solutions.
            (We use the majority to account for some random error exactly as FH does using the MSGlen.lenhypoThresh)

            :param solutionAcceptable: Dict of which solution is acceptable for which combination of messages.
            :return: Whether the given candidate is acceptable.
            """
            acceptCount = Counter(solutionAcceptable.values())
            mostlyAcceptable = bool(acceptCount[True] / len(acceptCount) > MSGlen.lenhypoThresh)
            # noinspection PyTypeChecker
            constantMultiplicator = all(numpy.round(Xarray[0,0], 8) == numpy.round(Xarray[1:,0], 8))
            logging.getLogger(__name__).debug(f"Candidate mostlyAcceptable {mostlyAcceptable} "
                                              f"and has constantMultiplicator {constantMultiplicator}.")
            return mostlyAcceptable and constantMultiplicator


class CategoricalCorrelatedField(FieldType,ABC):
    """
    Abstract class for inferring field types using categorical correlation of n-gram values with external values, e. g.,
    environmental information like addresses from encapsulation.
    """
    correlationThresh = 0.9  # 0.9, threshold for correlation between host ID and IP address(es) (FH, Sec. 3.2.3)
    minLenThresh = 4  # host ID fields must at least be 4 bytes long (FH, Sec. 3.2.3)
    n = 1  # Host-ID uses 8-bit/1-byte n-grams according to FH, Sec. 3.1.2, but this does not work well (see below)

    @classmethod
    @abstractmethod
    def _values2correlate2(cls, messages: List[L2NetworkMessage]):
        """
        Implement to determine the external values to correlate the n-grams of messages with.

        :param messages: Messages for which to generate correlation values.
        :return: The list of values, one for each message in the given order, to correlate to.
        """
        raise NotImplementedError("Implement this abstract class method in a subclass.")

    @classmethod
    def correlate(cls, messages: List[L2NetworkMessage]):
        """
        Generate n-grams at the same offsets for each message an correlate each n-gram using
        categorical correlation: R(x, y) = I(x: y)/H(x, y) \in [0,1]
        Uses cls#n to determine the n-gram sizes and cls#_values2correlate2() to obtain tuples of data to correlate.

        :param messages: Messages to generate n-grams to correlate to.
        :return: Correlation values for each offset of n-grams generated from the messages.
        """
        # ngram at offset and src address
        ngramsSrcs = list()
        categoricalCorrelation = list()
        corrValues = cls._values2correlate2(messages)
        # Iterate n-grams of all messages
        for ngrams in zip(*(NgramIterator(msg, n=CategoricalCorrelatedField.n) for msg in messages)):
            ngSc = numpy.array([intsFromNgrams(ngrams), corrValues])
            # categoricalCorrelation: R(x, y) = I(x: y)/H(x, y) \in [0,1]
            catCorr = drv.information_mutual(ngSc[0], ngSc[1]) / drv.entropy_joint(ngSc)
            ngramsSrcs.append(ngSc)
            categoricalCorrelation.append(catCorr)
        return categoricalCorrelation

    @classmethod
    def catCorrPosLen(cls, categoricalCorrelation: List[float]):
        """
        Merge consecutive candidate n-grams with categoricalCorrelation > correlationThresh.
        Filters n-gram offsets on defined thresholds (FH, Sec. 3.2.3) by their categorical correlation values to
            * correlation between host ID and IP address(es) > correlationThresh
            * discard short fields < minHostLenThresh

        :param categoricalCorrelation: Correlation values for each offset of n-grams generated from the messages.
        :return: List of start-length tuples with categorical correlation above threshold and not being a short field.
        """
        catCorrOffsets = [ offset for offset, catCorr in enumerate(categoricalCorrelation)
                           if catCorr > cls.correlationThresh ]
        catCorrRanges = list2ranges(catCorrOffsets)
        # discard short fields < minHostLenThresh
        return [ (start, length) for start, length in catCorrRanges if length >= cls.minLenThresh ]

    @property
    def categoricalCorrelation(self):
        # !! This attribute needs to be defined in subclass init !!
        # noinspection PyUnresolvedReferences
        return self._categoricalCorrelation


class HostID(CategoricalCorrelatedField):
    """
    Host identifier (Host-ID) inference (FH, Sec. 3.2.3)
    Find n-gram that is strongly correlated with IP address of sender.

    # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
    # # # # Investigate low categoricalCorrelation for all but one byte within an address field (see NTP and DHCP).
    # # # # According to NTP offset 12 (REF ID, often DST IP address) and DHCP offsets (12, 17, and) 20 (IPs)
    # # # # this works in principle, but if the n-gram is too short the correlation gets lost for some n-grams.
    # # print(tabulate(zip(*[hostidfields.categoricalCorrelation]), showindex="always"))
    # # from matplotlib import pyplot
    # # pyplot.bar(range(len(hostidfields.categoricalCorrelation)), hostidfields.categoricalCorrelation)
    # # pyplot.show()
    # # # sum([msg.data[20:24] == bytes(map(int, msg.source.rpartition(':')[0].split('.'))) for msg in messages])
    # # # sum([int.from_bytes(messages[m].data[20:24], "big") == srcs[m] for m in range(len(messages))])
    # # # # While the whole bootp.ip.server [20:24] correlates nicely to the IP address, single n-grams don't.
    # # serverIP = [(int.from_bytes(messages[m].data[20:24], "big"), srcs[m]) for m in range(len(messages))]
    # # serverIP0 = [(messages[m].data[20], srcs[m]) for m in range(len(messages))]
    # # serverIP1 = [(messages[m].data[21], srcs[m]) for m in range(len(messages))]
    # # serverIP2 = [(messages[m].data[22], srcs[m]) for m in range(len(messages))]
    # # serverIP3 = [(messages[m].data[23], srcs[m]) for m in range(len(messages))]
    # # # nsp = numpy.array([sip for sip in serverIP])
    # # # # The correlation is perfect, if null values are omitted
    # # nsp = numpy.array([sip for sip in serverIP if sip[0] != 0])   #  and sip[0] == sip[1]
    # # # nsp0 = numpy.array(serverIP0)
    # # # nsp1 = numpy.array(serverIP1)
    # # # nsp2 = numpy.array(serverIP2)
    # # # nsp3 = numpy.array(serverIP3)
    # # nsp0 = numpy.array([sip for sip in serverIP0 if sip[0] != 0])
    # # nsp1 = numpy.array([sip for sip in serverIP1 if sip[0] != 0])
    # # nsp2 = numpy.array([sip for sip in serverIP2 if sip[0] != 0])
    # # nsp3 = numpy.array([sip for sip in serverIP3 if sip[0] != 0])
    # # for serverSrcPairs in [nsp, nsp0, nsp1, nsp2, nsp3]:
    # #     print(drv.information_mutual(serverSrcPairs[:, 0], serverSrcPairs[:, 1]) / drv.entropy_joint(serverSrcPairs.T))
    # # # # TODO This is no implementation error, but raises doubts about the Host-ID description completeness:
    # # # #  Probably it does not mention an Entropy filter, direction separation, or - most probably -
    # # # #  an iterative n-gram size increase (see MSGlen).
    # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #

    """
    typelabel = 'Host-ID'

    def __init__(self, messages: List[L2NetworkMessage]):
        super().__init__()
        self._messages = messages
        self._categoricalCorrelation = type(self).correlate(messages)
        self._catCorrPosLen = type(self).catCorrPosLen(self._categoricalCorrelation)
        self._segments = type(self)._posLen2segments(messages, self._catCorrPosLen)

    @classmethod
    def _values2correlate2(cls, messages: List[L2NetworkMessage]):
        """
        Recover byte representation of ipv4 address from Netzob message and make one int out if each.
        :param messages: Messages to generate n-grams to correlate to.
        :return:
        """
        return intsFromNgrams([bytes(map(int, msg.source.rpartition(':')[0].split('.'))) for msg in messages])


class SessionID(CategoricalCorrelatedField):
    """
    Session identifier (Session-ID) inference (FH, Section 3.2.4)
    Find n-gram that is strongly correlated with IP addresses of sender and receiver
    using categorical correlation like Host-ID.

    Most of FH, Section 3.2.4, refers to Host-ID, so we use all missing details from there and reuse the implementation.
    The only difference are the values to correlate (see #_values2correlate2())

    # # TODO Problem similar to Host-ID leads to same bad quality.
    # # Moreover, Host-ID will always return a subset of Session-ID fields, so Host-ID should get precedence.
    """
    typelabel = 'Session-ID'

    def __init__(self, messages: List[L2NetworkMessage]):
        super().__init__()
        self._messages = messages
        # #correlate() uses #_values2correlate2() to determine what to correlate.
        #   Thus is iterates the n-grams (n=1) and create (n-grams,(source,destination))-tuples
        #   (instead of just n-grams to source for Host-ID).
        # It correlates the n-grams to the (client IP, server IP) tuple by calculating the catCorr for the
        #   n-gram and the source/destination tuple
        self._categoricalCorrelation = type(self).correlate(messages)
        self._catCorrPosLen = type(self).catCorrPosLen(self._categoricalCorrelation)
        self._segments = type(self)._posLen2segments(messages, self._catCorrPosLen)

    @classmethod
    def _values2correlate2(cls, messages: List[L2NetworkMessage]):
        """
        Get source AND destination addresses in same manner as (just) source for Host-ID.
        Recover byte representation of ipv4 address from Netzob message and make one int out if each.

        :param messages: Messages to generate n-grams to correlate to.
        :return: integer representation of source and destination addresses for each message.
        """
        return intsFromNgrams([bytes(map(int,
                                         msg.source.rpartition(':')[0].split('.') +
                                         msg.destination.rpartition(':')[0].split('.'))
                                     ) for msg in messages])


class TransID(FieldType):
    """
    Transaction identifier (Trans-ID) inference (FH, Section 3.2.5, Fig. 3 right)
    """
    typelabel = 'Trans-ID'

    transSupportThresh = 0.8  # enough support in conversations (FH, Sec. 3.2.5)
    minFieldLength = 2  # merged n-grams must at least be this amount of bytes long
    # n-gram size is not explicitly given in FH, but the description (merging, sharp drops in entropy in Fig. 6)
    #   leads to assuming it should be 1.
    n = 1
    entropyThresh = 0.8  # Value not given in FH!
    # entropy in c2s/s2c + flows: threshold for high entropy is not given in FH! Use value determined
    #   by own empirics in base.entropyThresh

    def __init__(self, flows: Flows):
        super().__init__()

        # prepare instance attributes
        self._flows = flows
        self._c2s, self._s2c = self._flows.splitDirections()  # type: List[L4NetworkMessage], List[L4NetworkMessage]
        self._c2sEntropyFiltered = None
        self._s2cEntropyFiltered = None
        self._c2sConvsEntropyFiltered = dict()
        self._s2cConvsEntropyFiltered = dict()
        self._c2sHorizontalOffsets = None
        self._s2cHorizontalOffsets = None
        self._c2sCombinedOffsets = None
        self._s2cCombinedOffsets = None
        self._valuematch = dict()
        # noinspection PyTypeChecker
        self._c2sConsistentRanges = None  # type: Iterable[Tuple[int, int]]
        # noinspection PyTypeChecker
        self._s2cConsistentRanges = None  # type: Iterable[Tuple[int, int]]

        # Infer
        self._verticalAndHorizontalRandomNgrams()
        self._constantQRvalues()
        self._consistentCandidates()
        # TODO not needed for textual protocols (FH, Sec. 3.2.5, last sentence)
        self._c2sConsistentRanges = type(self)._mergeAndFilter(self._c2sConsistentCandidates)
        self._s2cConsistentRanges = type(self)._mergeAndFilter(self._s2cConsistentCandidates)
        self._segments = \
            type(self)._posLen2segments(self._c2s, self._c2sConsistentRanges) + \
            type(self)._posLen2segments(self._s2c, self._s2cConsistentRanges)

    @classmethod
    def entropyFilteredOffsets(cls, messages: List[AbstractMessage], absolute=True):
        """
        Find offsets of n-grams (with the same offset in different messages of the list), that are random,
        i. e., that have a entropy > x (threshold)

        FH, Section 3.2.5

        :param messages: Messages to generate n-grams from
        :param absolute: Use the absolute constant for the threshold if true,
            make it relative to the maximum entropy if False.
        :return: Returns a list of offsets that have non-constant and non-random (below entropyThresh) entropy.
        """
        entropy = pyitNgramEntropy(messages, cls.n)
        entropyThresh = cls.entropyThresh if absolute else max(entropy) * cls.entropyThresh
        return [offset for offset, entropy in enumerate(entropy) if entropy > entropyThresh]

    def _verticalAndHorizontalRandomNgrams(self):
        """
        Determine n-grams that are "random across vertical and horizontal collections" (FH, Sec. 3.2.5).

        Output is written to self._c2sCombinedOffsets and self._s2cCombinedOffsets.
        Moreover, intermediate results are persisted in instance attributes for evaluation.
        """
        # vertical collections
        c2s, s2c = self._flows.splitDirections()  # type: List[L4NetworkMessage], List[L4NetworkMessage]
        self._c2sEntropyFiltered = type(self).entropyFilteredOffsets(c2s)
        self._s2cEntropyFiltered = type(self).entropyFilteredOffsets(s2c)
        # print('_c2sEntropyFiltered')
        # pprint(self._c2sEntropyFiltered)
        # print('_s2cEntropyFiltered')
        # pprint(self._s2cEntropyFiltered)

        # # horizontal collections: intermediate entropy of n-grams for debugging
        # self._c2sConvsEntropy = dict()
        # for key, conv in self._flows.c2sInConversations().items():
        #     self._c2sConvsEntropy[key] = pyitNgramEntropy(conv, type(self).n)
        # self._s2cConvsEntropy = dict()
        # for key, conv in self._flows.s2cInConversations().items():
        #     self._s2cConvsEntropy[key] = pyitNgramEntropy(conv, type(self).n)
        # print('_c2sConvsEntropy')
        # pprint(self._c2sConvsEntropy)
        # print('_s2cConvsEntropy')
        # pprint(self._s2cConvsEntropy)

        # horizontal collections: entropy of n-gram per the same offset in all messages of one flow direction
        for key, conv in self._flows.c2sInConversations().items():
            # The entropy is too low if the number of specimens is low -> relative to max
            # and ignore conversations of length 1
            # (TODO FH does not specify, but probably require even longer conversations to observe
            #   that the ID changes for each request/reply pair?
            #   "Transaction ID" in DHCP is a FP, since it is actually a Session-ID)
            if len(conv) <= 1:
                continue
            self._c2sConvsEntropyFiltered[key] = type(self).entropyFilteredOffsets(conv, False)
        for key, conv in self._flows.s2cInConversations().items():
            # The entropy is too low if the number of specimens is low -> relative to max
            # and ignore conversations of length 1
            # (TODO FH does not specify, but probably require even longer conversations to observe
            #   that the ID changes for each request/reply pair?
            #   "Transaction ID" in DHCP is a FP, since it is actually a Session-ID)
            if len(conv) <= 1:
                continue
            self._s2cConvsEntropyFiltered[key] = type(self).entropyFilteredOffsets(conv, False)
        # print('_c2sConvsEntropyFiltered')
        # pprint(_c2sConvsEntropyFiltered)
        # print('_s2cConvsEntropyFiltered')
        # pprint(_s2cConvsEntropyFiltered)

        # intersection of all c2s and s2c filtered offset lists (per flow)
        c2sOffsetLists = [set(offsetlist) for offsetlist in self._c2sConvsEntropyFiltered.values()]
        self._c2sHorizontalOffsets = set.intersection(*c2sOffsetLists) if len(c2sOffsetLists) > 0 else set()
        s2cOffsetLists = [set(offsetlist) for offsetlist in self._s2cConvsEntropyFiltered.values()]
        self._s2cHorizontalOffsets = set.intersection(*s2cOffsetLists) if len(s2cOffsetLists) > 0 else set()
        # offsets in _c2sEntropyFiltered where the offset is also in all of the lists of _c2sConvsEntropyFiltered
        # (TODO alternatively, deviating from FH, use the offset for each query specifically?)
        self._c2sCombinedOffsets = self._c2sHorizontalOffsets.intersection(self._c2sEntropyFiltered)
        # offsets in _c2sEntropyFiltered where the offset is also in all of the lists of _s2cConvsEntropyFiltered
        # (TODO alternatively, deviating from FH, use the entry for each response specifically?)
        self._s2cCombinedOffsets = self._s2cHorizontalOffsets.intersection(self._s2cEntropyFiltered)

    def _constantQRvalues(self):
        """
        Reqest/Response pairs: search for n-grams with constant values (differing offsets allowed)

        Output is placed in self._valuematch.
        """
        # compute Q->R association
        mqr = self._flows.matchQueryResponse()
        # from the n-gram offsets that passed the entropy-filters determine those that have the same value in mqr pairs
        for query, resp in mqr.items():
            qrmatchlist = self._valuematch[(query, resp)] = list()
            # value in query at any of the offsets in _c2sCombinedOffsets
            for c2sOffset in self._c2sCombinedOffsets:
                if len(query.data) < c2sOffset + type(self).n:
                    continue
                qvalue = query.data[c2sOffset:c2sOffset + type(self).n]
                # matches a value of resp at any of the offsets in _s2cCombinedOffsets
                for s2cOffset in self._s2cCombinedOffsets:
                    if len(resp.data) < s2cOffset + type(self).n:
                        continue
                    rvalue = resp.data[s2cOffset:s2cOffset + type(self).n]
                    if qvalue == rvalue:
                        qrmatchlist.append((c2sOffset, s2cOffset))

    def _consistentCandidates(self):
        """
        measure consistency: offsets recognized in more than transSupportThresh of conversations

        Output is written to self._c2sConsistentCandidates and self._s2cConsistentCandidates
        """
        c2sCandidateCount = Counter()
        s2cCandidateCount = Counter()
        for offsetlist in self._valuematch.values():        # (query, resp), offsetlist
            if len(offsetlist) < 1:
                continue
            # transpose to offsets per direction
            c2sOffsets, s2cOffsets = zip(*offsetlist)
            c2sCandidateCount.update(set(c2sOffsets))
            s2cCandidateCount.update(set(s2cOffsets))
        self._c2sConsistentCandidates = [offset for offset, cc in c2sCandidateCount.items() if
                                   cc > type(self).transSupportThresh * len(self._c2s)]
        self._s2cConsistentCandidates = [offset for offset, cc in s2cCandidateCount.items() if
                                   cc > type(self).transSupportThresh * len(self._s2c)]

    @classmethod
    def _mergeAndFilter(cls, consistentCandidates):
        """
        merge and filter candidates by minimum length
        """
        return [ol for ol in list2ranges(consistentCandidates) if ol[1] >= cls.minFieldLength]


# Host-ID will always return a subset of Session-ID fields, so Host-ID should get precedence
# MSG-Len would be overwritten by MSG-Type (see SMB: nbss.length), so first use MSG-Len
precedence = {MSGlen.typelabel: 0, MSGtype.typelabel: 1, HostID.typelabel: 2,
              SessionID.typelabel: 3, TransID.typelabel: 4, Accumulator.typelabel: 5}
"""
The order in which to map field types to messages. 
Lower numbers take precedence over higher numbers, so that the type with the higher number will be ignored 
if overlapping at the same offet range in the message.
"""