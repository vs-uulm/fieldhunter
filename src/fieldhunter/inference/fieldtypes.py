"""
Infer message field types according to the FieldHunter paper Section 3.2
"""
from typing import List, Tuple, Dict, Iterable
import random
from itertools import groupby, product, chain, combinations
from collections import Counter
from abc import ABC, abstractmethod

import numpy
from scipy.stats import pearsonr
from pyitlib import discrete_random_variable as drv
from netzob.Model.Vocabulary.Messages.AbstractMessage import AbstractMessage
from netzob.Model.Vocabulary.Messages.L2NetworkMessage import L2NetworkMessage
from netzob.Model.Vocabulary.Messages.L4NetworkMessage import L4NetworkMessage

from fieldhunter.utils.base import qrAssociationCorrelation, verticalByteMerge, mutualInformationNormalized, \
    list2ranges, Flows, NgramIterator, iterateSelected, intsFromNgrams, \
    ngramIsOverlapping, pyitNgramEntropy
from nemere.inference.analyzers import Value
from nemere.inference.segments import TypedSegment




class FieldType(ABC):
    typelabel = None

    def __init__(self):
        self._segments = list()

    @property
    def segments(self) -> List[List[TypedSegment]]:
        """
        :return: Final result as segments that are of the inferred type.
        """
        return self._segments

    @classmethod
    def _posLen2segments(cls, messages: List[L2NetworkMessage], posLen: Iterable[Tuple[int, int]]) \
            -> List[List[TypedSegment]]:
        """
        Generate Segments from remaining field ranges.
        :param messages: Messages to generate n-grams to correlate to.
        :param posLen: List of start-length tuples to create messages from.
        :return: Lists of segments per message generated from the posLen parameter.
        """
        segments = list()
        for message in messages:
            mval = Value(message)
            segs4msg = list()
            for start, length in posLen:
                # check if boundaries fit into message
                if start + length <= len(mval.values):
                    segs4msg.append(TypedSegment(mval, start, length, cls.typelabel))
            segments.append(segs4msg)
        return segments


class NonConstantNonRandomEntropyFieldType(FieldType, ABC):
    # constant entropyThresh: value 0.4 determined by own empirics (notes about that?)
    entropyThresh = 0.4  # Value not given in FH!

    @classmethod
    def entropyFilteredOffsets(cls, messages: List[AbstractMessage], n: int):
        """
        Find offsets of n-grams (with the same offset in different messages of the list), that are not constant and not
        random, i. e., that have a entropy > 0 and < x (threshold)

        FH, Section 3.2.1

        :param messages: Messages to generate n-grams from
        :param n: The $n$ in n-gram
        :return: Returns a list of offsets that have non-constant and non-random (below entropyThresh) entropy.
        """
        entropy = pyitNgramEntropy(messages, n)
        return [offset for offset, entropy in enumerate(entropy) if 0 < entropy < cls.entropyThresh]


class MSGtype(NonConstantNonRandomEntropyFieldType):
    """
    Message type (MSG-Type) inference (FH, Section 3.2.1, Fig. 3 left).

    Properties enable access to intermediate and final results.
    """
    typelabel = "MSG-Type"
    causalityThresh = 0.6  # FH, Sec. 3.2.1 says 0.8, but that leaves no candidates for our traces

    def __init__(self, flows: Flows):
        super().__init__()

        c2s, s2c = flows.splitDirections()  # type: List[L4NetworkMessage], List[L4NetworkMessage]
        # pprint(c2s)
        # pprint(s2c)

        # # alternative entropy calculation
        # c2sEntropy = entropyVertical(c2s)
        # s2cEntropy = entropyVertical(s2c)
        # print(tabulate(zip(c2sEntropy, s2cEntropy), headers=["c2s", "s2c"], showindex=True))

        # discard constant and random offsets
        self._c2sEntropyFiltered = type(self).entropyFilteredOffsets(c2s, 1)
        self._s2cEntropyFiltered = type(self).entropyFilteredOffsets(s2c, 1)
        # print(c2sEntropyFiltered)
        # print(s2cEntropyFiltered)

        # from collections import Counter
        # print(tabulate(Counter(msg.data[2:4].hex() for msg in c2s).most_common()))
        # print(tabulate(Counter(msg.data[2:4].hex() for msg in s2c).most_common()))

        # compute Q->R association
        mqr = flows.matchQueryRespone()
        # Mutual information
        self._qrCausality = qrAssociationCorrelation(mqr)
        # filter: only if offset is in c2sEntropyFiltered/s2cEntropyFiltered and the causality is greater than the causalityThresh
        self._filteredCausality = {offset: self.qrCausality[offset] for offset in
                             set(self.c2sEntropyFiltered).intersection(self.s2cEntropyFiltered)
                             if self.qrCausality[offset] > MSGtype.causalityThresh}
        # filteredCausality are offsets of MSG-Type candidate n-grams
        # print(tabulate(sorted(filteredCausality.items())))

        # Merge n-grams above causality threshold and check correlation
        self._mergingOffsets = list()
        for offset in sorted(self.filteredCausality.keys()):
            self._mergingOffsets.append(offset)
            qMergedField, rMergedField = verticalByteMerge(mqr, self.offsets)
            self._mergedCausality = mutualInformationNormalized(qMergedField, rMergedField)
            if self._mergedCausality <= MSGtype.causalityThresh:
                # Filter problematic n-grams
                self._mergingOffsets.pop()
        # re-calculate in case the last iteration removed a problematic n-gram
        qMergedField, rMergedField = verticalByteMerge(mqr, self.offsets)
        self._mergedCausality = mutualInformationNormalized(qMergedField, rMergedField)

        # print("mergedCausality", mergedCausality)
        # print("mergingOffsets", mergingOffsets)
        # print("  from offsets", sorted(filteredCausality.keys()))

        # create segments from bytes in mergingOffsets
        self._msgtypeRanges = list2ranges(self.offsets)
        self._segments = type(self)._posLen2segments(c2s + s2c, self._msgtypeRanges)


    @property
    def s2cEntropyFiltered(self) -> List[int]:
        """
        :return: The offsets for which the vertical entropies of all the server to client messages is
            greater than zero and less than MSGtype.entropyThresh
        """
        return self._s2cEntropyFiltered

    @property
    def c2sEntropyFiltered(self) -> List[int]:
        """
        :return: The offsets for which the vertical entropies of all the client to server messages is
            greater than zero and less than MSGtype.entropyThresh
        """
        return self._c2sEntropyFiltered

    @property
    def qrCausality(self) -> Dict[int,float]:
        return self._qrCausality

    @property
    def filteredCausality(self) -> Dict[int,float]:
        return self._filteredCausality

    @property
    def mergedCausality(self) -> List[int]:
        return self._mergedCausality

    @property
    def offsets(self):
        """
        :return: Final result as individual byte offsets of offsets that are MSG-Types
        """
        return self._mergingOffsets

    @property
    def ranges(self) -> List[Tuple[int, int]]:
        """
        :return: Final result as ranges of offsets that are MSG-Types
        """
        return self._msgtypeRanges


class MSGlen(NonConstantNonRandomEntropyFieldType):
    """
    Message length (MSG-Len) inference (FH, Section 3.2.2, Fig. 3 center).
    Application message length, linearly correlates with message size.

    Properties enable access to intermediate and final results.
    """
    typelabel = "MSG-Len"
    # coefficient threshold 0.6 (FH, Section 3.2.2)
    minCorrelation = 0.6
    # MSG-Len hypothesis threshold 0.9 (FH, Section 3.2.2)
    lenhypoThresh = 0.9

    def __init__(self, flows: Flows):
        super().__init__()

        self._msgDirection = list()
        c2s, s2c = flows.splitDirections()  # type: List[L4NetworkMessage], List[L4NetworkMessage]
        for direction in [c2s, s2c]:  # per direction - for MSG-Len this is pointless, but the paper says to do it.
            self._msgDirection.append(MSGlen.Direction(direction))

    @property
    def acceptedCandidatesPerDir(self) -> List[Dict[int, int]]:
        return [mldir.acceptedCandidates for mldir in self._msgDirection]

    @property
    def segments(self) -> List[List[TypedSegment]]:
        return list(chain.from_iterable([mldir.segments for mldir in self._msgDirection]))

    class Direction(object):
        """
        Encapsulates direction-wise inference of fields.
        Roughly corresponds to the S2C-collection branch depicted in the flow graph of FH, Fig. 3 center.

        Provides methods to extract different size collections, finding candidates by Pearson correlation coefficient,
        and verifying the hypothesis of candidates denoting the length of the message.
        """
        def  __init__(self, direction: List[L4NetworkMessage]):
            self._direction = direction
            # noinspection PyTypeChecker
            self._msgbylen = None  # type: Dict[int, List[L4NetworkMessage]]
            """Homogeneous Size Collections"""
            # noinspection PyTypeChecker
            self._msgmixlen = None  # type: List[L4NetworkMessage]
            # noinspection PyTypeChecker
            self._candidateAtNgram = None  # type: Dict[int, List[int]]
            # noinspection PyTypeChecker
            self._acceptedCandidates = None  # type: Dict[int, int]
            """Associates offset with a field length (n-gram's n) to define a list of unambiguous MSG-Len candidates"""
            # noinspection PyTypeChecker
            self._acceptedX = None  # type: Dict[int, numpy.ndarray]
            # # noinspection PyTypeChecker
            # self._segments = list()  # type: List[List[TypedSegment]]

            self.differentSizeCollections()
            self.findCandidates()
            self.verifyCandidates()
            # # validation of TODO in #verifyCandidates for SMB [1., 4.] -> would help
            # aX = numpy.array(self._acceptedX[0])
            # aX.mean(0)
            # aX.min(0)
            # aX.max(0)

            # create segments for each accepted candidate
            self._segments = MSGlen._posLen2segments(self._direction, self.acceptedCandidates.items())

        def differentSizeCollections(self):
            """
            "stratifying messages by length": extract different size collection -> vector of message lengths

            :return: List of messages that contains an equal amount of messages of each length,
                List of according message lengths
            """
            keyfunc = lambda m: len(m.data)
            # Homogeneous Size Collections
            self._msgbylen = {k: list(v) for k, v in groupby(sorted(self._direction, key=keyfunc), keyfunc)}
            minCollSize = min(len(v) for v in self._msgbylen.values())
            # generate size-heterogeneous collection by random sampling
            msgmixlen = list()
            for k, v in self._msgbylen.items():
                random.seed(42)
                if len(v) > minCollSize:
                    msgmixlen.extend(random.sample(v, k=minCollSize))
                else:
                    msgmixlen.extend(v)
            self._msgmixlen = msgmixlen

        def findCandidates(self):
            lens4msgmix = [len(m.data) for m in self._msgmixlen]  # type: List[int]
            candidateAtNgram = dict()
            # iterate n-grams' n=32, 24, 16 bits (4, 3, 2 bytes), see 3.1.2
            for n in [4, 3, 2]:
                # entropy filter for each n-gram offset -> field values matrix
                offsets = MSGlen.entropyFilteredOffsets(self._msgmixlen, n)
                # TODO currently only big endian, see #intsFromNgrams
                ngIters = (intsFromNgrams(iterateSelected(NgramIterator(msg, n), offsets)) for msg in self._msgmixlen)
                ngramsAtOffsets = numpy.array(list(ngIters))  # TODO check if null byte ngrams cause problems

                # correlate columns of ngramsAtOffsets to lens4msgmix
                pearsonAtOffset = list()
                for ngrams in ngramsAtOffsets.T:
                    # Pearson correlation coefficient (numeric value of n-gram) -> (len(msg.data))
                    pearsonAtOffset.append(pearsonr(ngrams, lens4msgmix)[0])
                candidateAtNgram[n] = [o for pao, o in zip(pearsonAtOffset, offsets) if pao > MSGlen.minCorrelation]
            self._candidateAtNgram = candidateAtNgram

        def verifyCandidates(self):
            """
            Verify length-hypothesis for candidates, solve for values at ngrams
            in candidateAtNgram (precedence for larger n).
              MSG_len = a * value + b (a > 0, b \in N)  - "Msg. Len. Model Parameters"
                  lens4msgmix = ngramsAtOffsets[:,candidateAtNgram[n]] * a + 1 * b
              threshold 0.9 of the message pairs with different lengths

            :return:
            """
            acceptedCandidates = dict()  # type: Dict[int, int]
            acceptedX = dict()
            #           specifying found acceptable solutions at offset (key) with n (value) for this direction
            for n in [4, 3, 2]:
                for offset in self._candidateAtNgram[n]:
                    # check precedence: if longer already-accepted n-gram overlaps this offset ignore
                    # noinspection PyTypeChecker
                    if not MSGlen.checkPrecedence(offset, n, acceptedCandidates.items()):
                        continue
                    # MSG-len hypothesis test - for ALL message pairs with different lengths (FH, 3.2.2 last paragraph)
                    #   - for the n-grams from this offset - keep only those offsets, where the threshold of pairs holds
                    solutionAcceptable = dict()  # type: Dict[Tuple[AbstractMessage, AbstractMessage], True]
                    Xes = list()
                    for l1, l2 in combinations(self._msgbylen.keys(),2):
                        for msg0, msg1 in product(self._msgbylen[l1], self._msgbylen[l2]):
                            ngramPair = [msg0.data[offset:offset + n], msg1.data[offset:offset + n]]
                            if ngramPair[0] == ngramPair[1]:
                                solutionAcceptable[(msg0, msg1)] = False
                                continue
                            A = numpy.array( [intsFromNgrams(ngramPair), [1, 1]] ).T
                            B = numpy.array([len(msg0.data), len(msg1.data)])
                            try:
                                X = numpy.linalg.inv(A).dot(B)
                                solutionAcceptable[(msg0, msg1)] = X[0] > 0 and X[1].is_integer()
                                Xes.append(X)
                            except numpy.linalg.LinAlgError:
                                print("LinAlgError occurred. Solution considered as non-acceptable.")
                                solutionAcceptable[(msg0, msg1)] = False
                    acceptCount = Counter(solutionAcceptable.values())
                    if acceptCount[True]/len(acceptCount) > MSGlen.lenhypoThresh:
                        acceptedCandidates[offset] = n
                        acceptedX[offset] = Xes
                        # TODO FH does not require this, but should not the values in A and B be equal within so that
                        #  a and b is one scalar value each?
            self._acceptedCandidates = acceptedCandidates
            self._acceptedX = {offset: numpy.array(aX) for offset, aX in acceptedX.items()}

        @property
        def acceptedCandidates(self) -> Dict[int, int]:
            """Associates offset with a field length (n-gram's n) to define a list of unambiguous MSG-Len candidates"""
            return self._acceptedCandidates

        @property
        def segments(self):
            return self._segments

    @staticmethod
    def checkPrecedence(offset: int, n: int, ngrams: Iterable[Tuple[int, int]]):
        """
        Has n-gram at offset precedence over all n-grams in ngrams?

        :param offset:
        :param n:
        :param ngrams: offset and n for a list of n-grams
        :return:
        """
        for o1, n1 in ngrams:
            if ngramIsOverlapping(offset, n, o1, n1):
                return False
        return True


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
        Uses cls.n to determine the n-gram sizes.

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
        # Attribute needs to be defined in subclass init.
        return self._categoricalCorrelation


class HostID(CategoricalCorrelatedField):
    """
    Host identifier (Host-ID) inference (FH, Sec. 3.2.3)
    Find n-gram that is strongly correlated with IP address of sender.
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
    """
    typelabel = 'Session-ID'

    def __init__(self, messages: List[L2NetworkMessage]):
        super().__init__()
        self._messages = messages
        # iterate the ngrams (n=1) and create a ngScDs (instead of just ngSc: ngram/source/destination)
        # correlate n-grams to (client IP, server IP) tuple by calculating the catCorr for the
        #   ngram and the source/destination tuple (TODO check out how to nest the tuple right)
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
        self._c2sConsistentRanges = None  # type: Iterable[Tuple[int, int]]
        self._s2cConsistentRanges = None  # type: Iterable[Tuple[int, int]]

        # Infer
        self._verticalAndHorizontalRandomNgrams()
        self._constantQRvalues()
        self._consistentCandidates()
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
            #  and ignore conversations of length 1 (TODO probably even more? "Transaction ID" in DHCP is a FP, since it is actually a Session-ID)
            if len(conv) <= 1:
                continue
            self._c2sConvsEntropyFiltered[key] = type(self).entropyFilteredOffsets(conv, False)
        for key, conv in self._flows.s2cInConversations().items():
            # The entropy is too low if the number of specimens is low -> relative to max
            #  and ignore conversations of length 1 (TODO probably even more? "Transaction ID" in DHCP is a FP, since it is actually a Session-ID)
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
        # (TODO use entry for this query specifically?)
        self._c2sCombinedOffsets = self._c2sHorizontalOffsets.intersection(self._c2sEntropyFiltered)
        # offsets in _c2sEntropyFiltered where the offset is also in all of the lists of _s2cConvsEntropyFiltered
        # (TODO the entry for this resp specifically?)
        self._s2cCombinedOffsets = self._s2cHorizontalOffsets.intersection(self._s2cEntropyFiltered)

    def _constantQRvalues(self):
        """
        Reqest/Response pairs: search for n-grams with constant values (differing offsets allowed)

        Output is placed in self._valuematch.
        """
        # compute Q->R association
        mqr = self._flows.matchQueryRespone()
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
        for (query, resp), offsetlist in self._valuematch.items():
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


class Accumulator(FieldType):
    """
    Accumulator inference (FH, Section 3.2.6)

    "Accumulators are fields that have increasing values over consecutive message within the same conversation."
        (FH, Sec. 3.2.6)
    """
    typelabel = 'Accumulator'

    endianness = 'big'
    ns = (8, 4, 3, 2)
    deltaEntropyThresh = 0.8  # Not given in FH, own empirics: 0.2

    def __init__(self, flows: Flows):
        super(Accumulator, self).__init__()

        # c2s and s2c independently
        self._c2sConvs = {key: list(sorted(conv, key=lambda m: m.date))
                          for key, conv in flows.c2sInConversations().items()}
        self._c2sDeltas = type(self).deltas(self._c2sConvs)
        self._c2sDeltaEntropies = type(self).entropies(self._c2sDeltas)

        self._s2cConvs = {key: list(sorted(conv, key=lambda m: m.date))
                          for key, conv in flows.s2cInConversations().items()}
        self._s2cDeltas = type(self).deltas(self._c2sConvs)
        self._s2cDeltaEntropies = type(self).entropies(self._s2cDeltas)

        # print('c2sDeltaEntropies (n: offset: value)')
        # pprint(c2sDeltaEntropies)
        # print('s2cDeltaEntropies (n: offset: value)')
        # pprint(s2cDeltaEntropies)

        c2s, s2c = flows.splitDirections()  # type: List[L4NetworkMessage], List[L4NetworkMessage]
        self._segments = self._posLen2segments(c2s, type(self).filter(self._c2sDeltaEntropies)) + \
                         self._posLen2segments(s2c, type(self).filter(self._s2cDeltaEntropies))

    @classmethod
    def deltas(cls, conversations: Dict[tuple, List[AbstractMessage]]) -> Dict[int, Dict[int, List[int]]]:
        """
        Value deltas per offset and n over all message-pairs of all conversations.

        :param conversations: Conversations need to be sorted in chronological order for the message pairs to produce
            meaningful deltas.
        :return: Pairwise deltas of values per offset and n-gram size.
        """
        deltas = dict()
        for key, conv in conversations.items():
            if len(conv) > 2:
                continue
            # subsequent messages per direction per conversation
            for msgA, msgB in zip(conv[:-1], conv[1:]):
                # iterate n-grams' n = 8, 4, 3, 2
                # combined from Sec. 3.1.2: n=32, 24, 16 bits (4, 3, 2 bytes)
                #       and see Sec. 3.2.6: n=64, 32, 16 bits (8, 4, 2 bytes)
                for n in cls.ns:
                    if n not in deltas:
                        deltas[n] = dict()
                    for offset, (ngramA, ngramB) in enumerate(zip(NgramIterator(msgA, n), NgramIterator(msgB, n))):
                        # calculate delta between n-grams (n and offset identical) two subsequent messages
                        # TODO also support little endian
                        delta = int.from_bytes(ngramB, cls.endianness) - int.from_bytes(ngramA, cls.endianness)
                        if offset not in deltas[n]:
                            deltas[n][offset] = list()
                        deltas[n][offset].append(delta)
        return deltas

    @classmethod
    def entropies(cls, deltas: Dict[int, Dict[int, List[int]]]) -> Dict[int, Dict[int, float]]:
        """
        For positive delta values with enough samples to calculate a meaningful entropy (>= 2),
        calculate the normalized entropies of the "compressed" (ln()) deltas.

        :param deltas: Pairwise deltas between values of subsequent messages in conversations
            at the same offset and with the same length (n): Dict[n, Dict[offset, delta] ].
        :return: Entropies of deltas per n-gram length and offset: Dict[n, Dict[offset, entropy] ].
        """
        lndeltas = dict()
        for n, offdel in deltas.items():
            lndeltas[n] = dict()
            for offset, dlts in offdel.items():
                # require more than 1 value to calculate a meaningful entropy
                if len(dlts) < 2:
                    continue
                npdlts = numpy.array(dlts)
                # require all deltas to be positive
                if any(npdlts <= 0):
                    continue
                # compress deltas by ln
                lndeltas[n][offset] = numpy.log(numpy.array(dlts))
        deltaEntropies = {n: {offset: drv.entropy(dlts)/numpy.log(n*8)
                              for offset, dlts in offdel.items()} for n, offdel in lndeltas.items()}
        return deltaEntropies

    @classmethod
    def filter(cls, deltaEntropies: Dict[int, Dict[int, float]]) -> List[Tuple[int, int]]:
        """
        Filter the entropies per n-gram size and offset to yield unambiguos candidates for accumulators.
        Filtering criteria are:
            * "fairly constant": relatively low entropy
            * previous filtering left over offsets for a n
            * prefer larger ns and smaller offsets if candidates are overlapping

        :param deltaEntropies: Entropies of deltas per n-gram length and offset: Dict[n, Dict[offset, entropy] ].
        :return: List of offsets and lengths that are valid field candidates.
        """
        # "fairly constant": relatively low entropy -> threshold (value not given in FH)
        filteredDE = {n: {offs: entr for offs, entr in offsdelt.items() if entr < cls.deltaEntropyThresh}
                         for n, offsdelt in deltaEntropies.items()}
        candidates = dict()  # type: Dict[int, List[int]]
        for n in reversed(sorted(filteredDE.keys())):
            # no offsets for this n-gram size
            if len(filteredDE[n]) == 0:
                continue
            for offset in sorted(filteredDE[n].keys()):
                # precedence for larger ns and smaller offsets: thats those we already found and added to candidates
                overlapps = False
                for candN, candOffs in candidates.items():
                    for candO in candOffs:
                        if ngramIsOverlapping(offset, n, candO, candN):
                            overlapps = True
                            break
                    if overlapps:
                        break
                if overlapps:
                    continue
                if not n in candidates:
                    candidates[n] = list()
                candidates[n].append(offset)
        posLen = [(o, n) for n, offsets in candidates.items() for o in offsets]
        return posLen


# Host-ID will always return a subset of Session-ID fields, so Host-ID should get precedence
precedence = {MSGtype.typelabel: 0, MSGlen.typelabel: 1, HostID.typelabel: 2,
                              SessionID.typelabel: 3, TransID.typelabel: 4, Accumulator.typelabel: 5}
