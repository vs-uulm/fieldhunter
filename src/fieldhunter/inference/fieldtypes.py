"""
Infer message field types according to the FieldHunter paper Section 3.2
"""

from typing import List, Tuple, Dict

from netzob.Model.Vocabulary.Messages.L4NetworkMessage import L4NetworkMessage

from fieldhunter.utils.base import pyitEntropyVertical, qrAssociationCorrelation, verticalByteMerge, mutualInformation, \
    list2ranges, Flows
from nemere.inference.analyzers import Value
from nemere.inference.segments import TypedSegment


class MSGtype(object):
    """
    Message type (MSG-Type) inference (FH, Section 3.2.1).

    Properties enable access to intermediate and final results.
    """

    entropyThresh = 0.4    # Not given in FH!
    causalityThresh = 0.6  # FH, Sec. 3.2.1 says 0.8, but that leaves no candidates for our traces

    def __init__(self, flows: Flows):
        c2s, s2c = flows.splitDirections()  # type: List[L4NetworkMessage], List[L4NetworkMessage]
        # pprint(c2s)
        # pprint(s2c)

        # # alternative entropy calculation
        # c2sEntropy = entropyVertical(c2s)
        # s2cEntropy = entropyVertical(s2c)
        # print(tabulate(zip(c2sEntropy, s2cEntropy), headers=["c2s", "s2c"], showindex=True))

        # discard constant and random offsets (threshold?)
        self._c2sEntropy = pyitEntropyVertical(c2s)
        self._c2sEntropyFiltered = [offset for offset, entropy
                                    in enumerate(self._c2sEntropy) if 0 < entropy < MSGtype.entropyThresh]
        self._s2cEntropy = pyitEntropyVertical(s2c)
        self._s2cEntropyFiltered = [offset for offset, entropy
                                    in enumerate(self._s2cEntropy) if 0 < entropy < MSGtype.entropyThresh]
        # print(tabulate(zip(c2sEntropy, s2cEntropy), headers=["c2s", "s2c"], showindex=True))
        # print(c2sEntropyFiltered)
        # print(s2cEntropyFiltered)

        # from collections import Counter
        # print(tabulate(Counter(msg.data[2:4].hex() for msg in c2s).most_common()))
        # print(tabulate(Counter(msg.data[2:4].hex() for msg in s2c).most_common()))

        # compute Q->R association/
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
            self._mergedCausality = mutualInformation(qMergedField, rMergedField)
            if self._mergedCausality <= MSGtype.causalityThresh:
                # Filter problematic n-grams
                self._mergingOffsets.pop()
        # re-calculate in case the last iteration removed a problematic n-gram
        qMergedField, rMergedField = verticalByteMerge(mqr, self.offsets)
        self._mergedCausality = mutualInformation(qMergedField, rMergedField)

        # print("mergedCausality", mergedCausality)
        # print("mergingOffsets", mergingOffsets)
        # print("  from offsets", sorted(filteredCausality.keys()))

        # create segments from bytes in mergingOffsets and compare to dissector/field type
        self._msgtypeRanges = list2ranges(self.offsets)
        self._msgtypeSegments = list()
        for message in c2s + s2c:
            segs4msg = list()
            for start, end in self._msgtypeRanges:
                segs4msg.append(TypedSegment(Value(message), start, end + 1 - start, "MSG-Type"))
            self._msgtypeSegments.append(segs4msg)


    @property
    def c2sEntropy(self) -> List[float]:
        """
        :return: The vertical entropies for each offset of all the client to server messages
        """
        return self._s2cEntropy

    @property
    def s2cEntropyFiltered(self) -> List[int]:
        """
        :return: The offsets for which the vertical entropies of all the server to client messages is
            greater than zero and less than MSGtype.entropyThresh
        """
        return self._s2cEntropyFiltered

    @property
    def s2cEntropy(self) -> List[float]:
        """
        :return: The vertical entropies for each offset of all the server to client messages
        """
        return self._c2sEntropy

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

    @property
    def segments(self) -> List[List[TypedSegment]]:
        """
        :return: Final result as segments that are MSG-Types
        """
        return self._msgtypeSegments
