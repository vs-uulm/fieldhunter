"""
Infer message field types exactly according to the FieldHunter paper Section 3.2

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

from nemere.inference.analyzers import Value
from nemere.inference.segments import TypedSegment
from tabulate import tabulate

from fieldhunter.utils.base import qrAssociationCorrelation, verticalByteMerge, mutualInformationNormalized, \
    list2ranges, Flows, NgramIterator, iterateSelected, intsFromNgrams, \
    ngramIsOverlapping, pyitNgramEntropy


# logging.getLogger(__name__).setLevel(logging.DEBUG)


class FieldType(ABC):
    """
    Generic, abstract base class for field types. Holds segments and a type label for the inferred fields.

    For DocTest example see fieldhunter.inference.common#segmentedMessagesAndSymbols()
    """
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
    def _posLen2segments(cls, messages: List[L2NetworkMessage],
                         posLen: Union[Iterable[Tuple[int, int]],ItemsView[int, int]]) \
            -> List[List[TypedSegment]]:
        """
        Generate Segments from (remaining) field ranges.

        For DocTest example see fieldhunter.inference.common#segmentedMessagesAndSymbols()

        :param messages: Messages to generate n-grams to correlate to.
        :param posLen: List of start-length tuples to create segments for from each message.
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
    """
    Abstract class for inferring field types using entropy of n-gram values
    where the entropy may neither be 0 (constant n-gram values)
    nor equal or greater than a threshold (random n-gram values).
    """
    # Value for entropyThresh not given in FH!
    # We use a constant entropyThresh of 0.4 determined by own empirics (results of /src/trace_statistics.py in
    #   nemesys-reports/NEMEFTR/fieldhunter/typeAndLengthEntropies.ods)
    entropyThresh = 0.4

    @classmethod
    def entropyFilteredOffsets(cls, messages: List[AbstractMessage], n: int):
        """
        Find offsets of n-grams (with the same offset in different messages of the list), that are not constant and not
        random, i. e., that have an entropy between and cls.entropyThresh.

        FH, Section 3.2.1

        :param messages: Messages to generate n-grams from
        :param n: The $n$ in n-gram
        :return: Returns a list of offsets that have non-constant and non-random (below entropyThresh) entropy.
        """
        entropy = pyitNgramEntropy(messages, n)
        ePo = [(offset, entropy) for offset, entropy in enumerate(entropy)]
        logging.getLogger(__name__).debug(f"Entropies per offset:\n{tabulate(ePo)}")
        return [offset for offset, entropy in enumerate(entropy) if 0 < entropy < cls.entropyThresh]


class MSGtype(NonConstantNonRandomEntropyFieldType):
    """
    Message type (MSG-Type) inference (FH, Section 3.2.1, Fig. 3 left).
    This type heuristic is based on the mutual information shared between n-grams at the same offset in the query and
    response messages. This assumes that message type fields are at the same position in query and response. Moreover,
    fields that solely denote whether a message is a query or a response, yield a undefined mutual information and thus
    cannot be detected as fields denoting a message type.

    The properties of this class provide access to intermediate and final results.
    """
    typelabel = "MSG-Type"
    causalityThresh = 0.8

    def __init__(self, flows: Flows):
        super().__init__()

        logger = logging.getLogger(__name__)
        c2s, s2c = flows.splitDirections()  # type: List[L4NetworkMessage], List[L4NetworkMessage]

        # discard constant and random offsets
        self._c2sEntropyFiltered = type(self).entropyFilteredOffsets(c2s, 1)
        self._s2cEntropyFiltered = type(self).entropyFilteredOffsets(s2c, 1)
        logger.info(f"c2sEntropyFiltered offsets: {self.c2sEntropyFiltered}")
        logger.info(f"s2cEntropyFiltered offsets: {self.s2cEntropyFiltered}")

        # compute Q->R association
        mqr = flows.matchQueryResponse()
        if len(mqr) < 2:
            # not enough query response pairs to continue analysis: create valid empty instance state and return
            self._qrCausality = dict()
            self._filteredCausality = dict()
            self._mergingOffsets = list()
            self._mergedCausality = dict()
            self._msgtypeRanges = list()
            self._segments = list()
            return
        # Mutual information
        self._qrCausality = qrAssociationCorrelation(mqr)
        # filter: only if offset is in c2sEntropyFiltered/s2cEntropyFiltered and the causality is greater than the causalityThresh
        self._filteredCausality = {offset: self.qrCausality[offset] for offset in
                             set(self.c2sEntropyFiltered).intersection(self.s2cEntropyFiltered)
                             if self.qrCausality[offset] > type(self).causalityThresh}
        # filteredCausality are offsets of MSG-Type candidate n-grams
        logger.info(f"filtered causality: {sorted(self.filteredCausality.items())}")

        # Merge n-grams above causality threshold and check correlation
        self._mergingOffsets = list()
        for offset in sorted(self.filteredCausality.keys()):
            self._mergingOffsets.append(offset)
            qMergedField, rMergedField = verticalByteMerge(mqr, self.offsets)
            mergedCausality = mutualInformationNormalized(qMergedField, rMergedField)
            if mergedCausality <= type(self).causalityThresh:
                # Filter problematic n-grams
                self._mergingOffsets.pop()
        # re-calculate in case the last iteration removed a problematic n-gram
        qMergedField, rMergedField = verticalByteMerge(mqr, self.offsets)
        self._mergedCausality = mutualInformationNormalized(qMergedField, rMergedField)
        logger.info(f"mergedCausality: {self.mergedCausality}")
        logger.info(f"  mergedOffsets: {self._mergingOffsets}")
        logger.info(f"   from offsets: {sorted(self.filteredCausality.keys())}")

        # create segments from bytes in mergingOffsets
        self._msgtypeRanges = list2ranges(self.offsets)
        self._segments = type(self)._posLen2segments(c2s + s2c, self._msgtypeRanges)


    @property
    def s2cEntropyFiltered(self) -> List[int]:
        """
        :return: The offsets for which the vertical entropies of all the server to client messages are
            greater than zero and less than MSGtype.entropyThresh
        """
        return self._s2cEntropyFiltered

    @property
    def c2sEntropyFiltered(self) -> List[int]:
        """
        :return: The offsets for which the vertical entropies of all the client to server messages are
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
    Find values in the message that linearly correlate with the application-layer message size.

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
        # per direction - for MSG-Len this is pointless, but the paper says to do it.
        #   it might rather be useful to separate message types (distinct formats) in this manner.
        for direction in [c2s, s2c]:
            self._msgDirection.append(type(self).Direction(direction))

    @property
    def acceptedCandidatesPerDir(self) -> List[Dict[int, int]]:
        return [mldir.acceptedCandidates for mldir in self._msgDirection]

    @property
    def segments(self) -> List[List[TypedSegment]]:
        return list(chain.from_iterable([mldir.segments for mldir in self._msgDirection]))

    class Direction(object):
        """
        Encapsulates direction-wise inference of MSGlen fields.
        Roughly corresponds to either the S2C-collection or C2S-collection branch
        depicted in the flow graph of FH, Fig. 3 center.

        Provides methods to extract different size collections, finding candidates by Pearson correlation coefficient,
        and verifying the hypothesis of candidates denoting the length of the message.
        """
        # TODO also support little endian (for our test traces, it was irrelevant)
        endianness = 'big'

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
            """Maps offsets to (a,b) that solve the linear equation in #verifyCandidates() 
            (FH: 'Msg. Len. Model Parameters')"""

            self.differentSizeCollections()
            self.findCandidates()
            self.verifyCandidates()

            # create segments for each accepted candidate
            self._segments = MSGlen._posLen2segments(self._direction, self.acceptedCandidates.items())

        def differentSizeCollections(self):
            """
            "stratifying messages by length": extract different size collection -> vector of message lengths

            :return: List of messages that contains an equal amount of messages of each length,
                i. e., List of according message lengths
            """
            if len(self._direction) == 0:  # "No flows in this direction."
                self._msgmixlen = list()
                return
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
            """
            Find message-length candidates (in parenthesis: block names from FH, Fig. 3 center):
            * filter for message offsets where the n-gram is not constant and not random (Entropy Filter)
            * correlate n-grams to message lengths (Pearson Correlation)

            :return: The offsets (dict value: list) where the Pearson Correlation
                exceeds the threshold MSGlen.minCorrelation
                for different sized n-grams (dict key).
            """
            # "Extract Vector of Message Length"
            lens4msgmix = [len(m.data) for m in self._msgmixlen]  # type: List[int]
            candidateAtNgram = dict()
            # iterate n-grams' n=32, 24, 16 bits (4, 3, 2 bytes), see 3.1.2
            for n in [4, 3, 2]:
                # entropy filter for each n-gram offset for "Field Values Matrix" below
                offsets = MSGlen.entropyFilteredOffsets(self._msgmixlen, n)
                # TODO currently only tested for big endian, see #intsFromNgrams
                # TODO for textual protocols decode the n-gram as (ASCII) number (FH, Sec. 3.2.2, second paragraph)
                ngIters = (intsFromNgrams(
                    iterateSelected(NgramIterator(msg, n), offsets), type(self).endianness) for msg in self._msgmixlen)
                # "Field Values Matrix"
                ngramsAtOffsets = numpy.array(list(ngIters))

                # correlate columns of ngramsAtOffsets to lens4msgmix
                pearsonAtOffset = list()
                for ngrams in ngramsAtOffsets.T:
                    # Pearson correlation coefficient (numeric value of n-gram) -> (len(msg.data))
                    pearsonAtOffset.append(pearsonr(ngrams, lens4msgmix)[0])
                candidateAtNgram[n] = [o for pao, o in zip(pearsonAtOffset, offsets) if pao > MSGlen.minCorrelation]
            self._candidateAtNgram = candidateAtNgram

        def verifyCandidates(self):
            """
            Verify the length-hypothesis for candidates, by solving the linear equation
            for values at the candidate n-grams in candidateAtNgram (precedence for larger n, i. e., longer fields):

              MSG_len = a * value + b (a > 0, b \in N)  - "Msg. Len. Model Parameters"
                  lens4msgmix = ngramsAtOffsets[:,candidateAtNgram[n]] * a + 1 * b

            At least a threshold 0.9 of the message pairs with different lengths has to fulfill the hypothesis.
            """
            acceptedCandidates = dict()  # type: Dict[int, int]
            acceptedX = dict()
            # specifying found acceptable solutions at offset (key) with n (value) for this direction
            for n in [4, 3, 2]:
                for offset in self._candidateAtNgram[n]:
                    # check precedence: if longer already-accepted n-gram overlaps this offset ignore
                    # noinspection PyTypeChecker
                    if not MSGlen.checkPrecedence(offset, n, acceptedCandidates.items()):
                        continue
                    # MSG-len hypothesis test - for ALL message pairs with different lengths (FH, 3.2.2 last paragraph)
                    #   - for the n-grams from this offset - keep only those offsets, where the threshold of pairs holds
                    solutionAcceptable = dict()  # type: Dict[Tuple[AbstractMessage, AbstractMessage], bool]
                    Xes = list()
                    for l1, l2 in combinations(self._msgbylen.keys(),2):
                        for msg0, msg1 in product(self._msgbylen[l1], self._msgbylen[l2]):
                            ngramPair = [msg0.data[offset:offset + n], msg1.data[offset:offset + n]]
                            if ngramPair[0] == ngramPair[1]:
                                solutionAcceptable[(msg0, msg1)] = False
                                continue
                            A = numpy.array( [intsFromNgrams(ngramPair), [1, 1]] ).T
                            B = numpy.array( [len(msg0.data), len(msg1.data)] )
                            try:  # solve the linear equation
                                X = numpy.linalg.inv(A).dot(B)
                                solutionAcceptable[(msg0, msg1)] = X[0] > 0 and X[1].is_integer()
                                Xes.append(X)
                            except numpy.linalg.LinAlgError:
                                print("LinAlgError occurred. Solution considered as non-acceptable.")
                                solutionAcceptable[(msg0, msg1)] = False
                    Xarray = numpy.array(Xes)
                    logging.getLogger(__name__).debug(f"Checking candidate with n = {n} at offset {offset}.")
                    if type(self)._candidateIsAcceptable(solutionAcceptable, Xarray):
                        acceptedCandidates[offset] = n
                        acceptedX[offset] = Xarray
            self._acceptedCandidates = acceptedCandidates
            self._acceptedX = acceptedX

        @staticmethod
        def _candidateIsAcceptable(solutionAcceptable: Dict[Tuple[AbstractMessage, AbstractMessage], bool],
                                   Xarray: numpy.ndarray):
            """
            Count the message pairs for which the solution is acceptable according to the MSG-len hypothesis test.

            :param solutionAcceptable: results of the the MSG-len hypothesis test for the Cartesian product of messages.
            :return: Whether this candidate is acceptable using MSGlen.lenhypoThresh.
            """
            acceptCount = Counter(solutionAcceptable.values())
            return bool(acceptCount[True]/len(acceptCount) > MSGlen.lenhypoThresh)

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

    CategoricalCorrelatedField#correlate() uses #_values2correlate2() to determine what to correlate.
    This is different for the subclasses (HostID, SessionID). It iterates the n-grams (n=1) and creates
    n-grams-to-source-IP-tuples for Host-ID or (n-grams,(source IP,destination IP))-tuples for Session-ID.
    It correlates the n-grams to the respective tuple by calculating the catCorr for the
    n-gram and the source/destination tuple.
    """
    correlationThresh = 0.9  # 0.9, threshold for correlation between host ID and IP address(es) (FH, Sec. 3.2.3)
    minLenThresh = 4  # host ID fields must at least be 4 bytes long (FH, Sec. 3.2.3)

    def __init__(self, messages: List[L4NetworkMessage]):
        super().__init__()
        self._messages = messages
        filteredMessages = type(self)._filterMessages(messages)
        # We correlate only the filtered messages...
        self._categoricalCorrelation = type(self).correlate(filteredMessages)
        self._catCorrPosLen = type(self).catCorrPosLen(self._categoricalCorrelation)
        # ... but use the positions to generate segments for all messages. This might not always be wise.
        self._segments = type(self)._posLen2segments(self._messages, self._catCorrPosLen)

    @classmethod
    def _filterMessages(cls, messages: List[L4NetworkMessage]):
        """
        Filter messages used to correlate in the first place. To be overwritten by subclasses.
        This basic implementation passes the input list unchanged.
        """
        return messages

    @classmethod
    @abstractmethod
    def _values2correlate2(cls, messages: List[L4NetworkMessage]) -> List[int]:
        """
        Implement to determine the external values to correlate the n-grams of messages with.

        :param messages: Messages for which to generate correlation values.
        :return: The list of values, one for each message in the given order, to correlate to.
        """
        raise NotImplementedError("Implement this abstract class method in a subclass.")

    @classmethod
    def correlate(cls, messages: List[L4NetworkMessage], n: int = 1):
        # noinspection PyShadowingNames
        r"""
        Generate n-grams at the same offsets for each message an correlate each n-gram using
        categorical correlation: R(x, y) = I(x: y)/H(x, y) \in [0,1]
        Uses cls#n to determine the n-gram sizes and cls#_values2correlate2() to obtain tuples of data to correlate.

        >>> from fieldhunter.inference.fieldtypes import SessionID
        >>> from netzob.Model.Vocabulary.Messages.L4NetworkMessage import L4NetworkMessage
        >>> messages = [
        ...     L4NetworkMessage(b"session111\x42\x17\x23\x00\x08\x15",
        ...         l3SourceAddress="1.2.3.100", l3DestinationAddress="1.2.3.1"),
        ...     L4NetworkMessage(b"session111xe4\x83\x82\x85\xbf",
        ...         l3SourceAddress="1.2.3.100", l3DestinationAddress="1.2.3.1"),
        ...     L4NetworkMessage(b"session111\x42\x17\xf9\x0b\x00b\x12O",
        ...         l3SourceAddress="1.2.3.100", l3DestinationAddress="1.2.3.1"),
        ...     L4NetworkMessage(b"session222\x42\x17Jk\x8a1e\xb5",
        ...         l3SourceAddress="1.2.3.2", l3DestinationAddress="1.2.3.100"),
        ...     L4NetworkMessage(b"session222L\xab\x83\x1a\xef\x13",
        ...         l3SourceAddress="1.2.3.2", l3DestinationAddress="1.2.3.100"),
        ... ]
        >>> SessionID.correlate(messages)
        [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 0.21851654863632566, 0.21851654863632566, 0.4181656600790516, 0.4181656600790516, 0.4181656600790516, 0.4181656600790516]

        :param messages: Messages to generate n-grams to correlate to.
        :param n: Host-ID uses 8-bit/1-byte n-grams according to FH, Sec. 3.1.2, but this does not work well
            (see fieldtypesRelaxed.CategoricalCorrelatedField)
        :return: Correlation values for each offset of n-grams generated from the messages.
        """
        # ngram at offset and src address
        ngramsSrcs = list()
        categoricalCorrelation = list()
        corrValues = cls._values2correlate2(messages)
        # Iterate n-grams of all messages
        for ngrams in zip(*(NgramIterator(msg, n=n) for msg in messages)):
            ngSc = cls._combineNgrams2Values(ngrams, corrValues)
            if ngSc.size > 0:
                # categoricalCorrelation: R(x, y) = I(x: y)/H(x, y) \in [0,1]
                catCorr = drv.information_mutual(ngSc[0], ngSc[1]) / drv.entropy_joint(ngSc)
            else:
                catCorr = numpy.nan
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
        # !! The attribute self._categoricalCorrelation needs to be defined in subclass init !!
        # noinspection PyUnresolvedReferences
        return self._categoricalCorrelation

    @classmethod
    def _combineNgrams2Values(cls, ngrams: Iterable[bytes], values: List[int]) -> numpy.ndarray:
        return numpy.array([intsFromNgrams(ngrams), values])

    @classmethod
    def _srcDstBytes(cls, messages: List[L4NetworkMessage]):
        return [ (
            bytes(map(int, msg.source.rpartition(':')[0].split('.'))),
            bytes(map(int, msg.destination.rpartition(':')[0].split('.')))
        ) for msg in messages]


class HostID(CategoricalCorrelatedField):
    """
    Host identifier (Host-ID) inference (FH, Sec. 3.2.3)
    Find n-gram that is strongly correlated with IP address of sender.
    """
    typelabel = 'Host-ID'

    @classmethod
    def _values2correlate2(cls, messages: List[L4NetworkMessage]):
        """
        Recover byte representations of the IPv4 addresses from all Netzob messages and make one int out if each.
        :param messages: Messages to generate n-grams to correlate to.
        :return:
        """
        return intsFromNgrams(src for src, dst in cls._srcDstBytes(messages))


class SessionID(CategoricalCorrelatedField):
    """
    Session identifier (Session-ID) inference (FH, Section 3.2.4)
    Find n-gram that is strongly correlated with IP addresses of sender and receiver
    using categorical correlation like Host-ID.

    Most of FH, Section 3.2.4, refers to Host-ID, so we use all missing details from there and reuse the implementation.
    The only difference are the values to correlate (see #_values2correlate2())
    """
    typelabel = 'Session-ID'

    @classmethod
    def _values2correlate2(cls, messages: List[L4NetworkMessage]):
        """
        Get source AND destination addresses in the same manner as (just) the source for Host-ID.
        Recover byte representations of the IPv4 addresses from all Netzob messages and make one int out if each.

        :param messages: Messages to generate n-grams to correlate to.
        :return: integer representation of source and destination addresses for each message.
        """
        return intsFromNgrams(src+dst for src, dst in cls._srcDstBytes(messages))


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
    entropyThresh = 0.6  # Value is not given in FH paper!
    """
    entropy in c2s/s2c + flows: threshold for high entropy is not given in FH!
    We use a value determined by own empirics: see entropy plots from src/trace_statistics.py
    """
    absoluteEntropy = True

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
        Find offsets of n-grams (with the same offset in different messages of the list) that are random,
        i. e., that have a entropy greater than entropyThresh (cls.entropyThresh or relative).

        FH, Section 3.2.5

        :param messages: Messages to generate n-grams from
        :param absolute: Use the absolute constant for the threshold if true,
            make it relative to the maximum entropy if False.
        :return: Returns a list of offsets that have non-constant and non-random (below entropyThresh) entropy.
        """
        if len(messages) > 0:
            entropy = pyitNgramEntropy(messages, cls.n)
            entropyThresh = cls.entropyThresh if absolute else max(entropy) * cls.entropyThresh
            return [offset for offset, entropy in enumerate(entropy) if entropy > entropyThresh]
        else:
            return []

    def _verticalAndHorizontalRandomNgrams(self):
        """
        Determine n-grams that are "random across vertical and horizontal collections" (FH, Sec. 3.2.5).

        Output is written to self._c2sCombinedOffsets and self._s2cCombinedOffsets.
        Moreover, intermediate results are persisted in instance attributes for evaluation.
        """
        logger = logging.getLogger(__name__)
        # vertical collections
        c2s, s2c = self._flows.splitDirections()  # type: List[L4NetworkMessage], List[L4NetworkMessage]
        self._c2sEntropyFiltered = type(self).entropyFilteredOffsets(c2s, type(self).absoluteEntropy)
        self._s2cEntropyFiltered = type(self).entropyFilteredOffsets(s2c, type(self).absoluteEntropy)
        logger.info(f"c2sEntropyFiltered offsets: {self._c2sEntropyFiltered}")
        logger.info(f"s2cEntropyFiltered offsets: {self._s2cEntropyFiltered}")

        # # DEBUGGING horizontal collections: intermediate entropy of n-grams
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
        #
        # horizontal collections: entropy of n-gram per the same offset in all messages of one flow direction
        self._c2sConvsEntropyFiltered = type(self)._horizontalRandomNgrams(
            self._flows.c2sInConversations(), self._c2sEntropyFiltered)
        self._s2cConvsEntropyFiltered = type(self)._horizontalRandomNgrams(
            self._flows.s2cInConversations(), self._s2cEntropyFiltered)
        logger.info('c2sConvsEntropyFiltered: ' + repr(self._c2sConvsEntropyFiltered.values()))
        logger.info('s2cConvsEntropyFiltered: ' + repr(self._s2cConvsEntropyFiltered.values()))

        # intersection of all c2s and s2c filtered offset lists (per flow)
        c2sOffsetLists = [set(offsetlist) for offsetlist in self._c2sConvsEntropyFiltered.values()]
        self._c2sHorizontalOffsets = set.intersection(*c2sOffsetLists) if len(c2sOffsetLists) > 0 else set()
        s2cOffsetLists = [set(offsetlist) for offsetlist in self._s2cConvsEntropyFiltered.values()]
        self._s2cHorizontalOffsets = set.intersection(*s2cOffsetLists) if len(s2cOffsetLists) > 0 else set()

        # offsets in _c2sEntropyFiltered where the offset is also in all of the lists of _c2sConvsEntropyFiltered
        self._c2sCombinedOffsets = self._c2sHorizontalOffsets.intersection(self._c2sEntropyFiltered)
        # offsets in _c2sEntropyFiltered where the offset is also in all of the lists of _s2cConvsEntropyFiltered
        self._s2cCombinedOffsets = self._s2cHorizontalOffsets.intersection(self._s2cEntropyFiltered)

    @classmethod
    def _horizontalRandomNgrams(cls, conversions: Dict[tuple, List[AbstractMessage]],
                                verticalEntropyFiltered: List[int]) -> Dict[Union[Tuple, None], List[int]]:
        """
        Filter in offsets that are random horizontally

        :param conversions:
        :param verticalEntropyFiltered:
        :return:
        """
        filteredOutput = dict()
        # horizontal collections: entropy of n-gram per the same offset in all messages of one flow direction
        for key, conv in conversions.items():
            filteredOutput[key] = cls.entropyFilteredOffsets(conv, cls.absoluteEntropy)
        return filteredOutput

    def _constantQRvalues(self):
        """
        Request/Response pairs: search for n-grams with constant values (differing offsets allowed)

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


class Accumulator(FieldType):
    """
    Accumulator inference (FH, Section 3.2.6)

    "Accumulators are fields that have increasing values over consecutive message within the same conversation."
        (FH, Sec. 3.2.6)
    """
    typelabel = 'Accumulator'

    # TODO also support little endian (for our test traces, it was irrelevant)
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
                        # TODO test support little endian (for our test traces, it was irrelevant)
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
# MSG-Len would be overwritten by MSG-Type (see SMB: nbss.length), so first use MSG-Len
precedence = {MSGlen.typelabel: 0, MSGtype.typelabel: 1, HostID.typelabel: 2,
              SessionID.typelabel: 3, TransID.typelabel: 4, Accumulator.typelabel: 5}
"""
The order in which to map field types to messages. 
Lower numbers take precedence over higher numbers, so that the type with the higher number will be ignored 
if overlapping at the same offet range in the message.
"""