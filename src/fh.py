"""
Only implements FH's binary message handling using n-grams (not textual using delimiters!)
"""

from argparse import ArgumentParser
from itertools import groupby, product, chain, combinations
from collections import Counter
from os.path import isfile, basename, splitext
import random, numpy
from typing import Dict, Tuple, Iterable, Sequence, List

from netzob.Model.Vocabulary.Messages.AbstractMessage import AbstractMessage
from scipy.stats import pearsonr

# noinspection PyUnresolvedReferences
from tabulate import tabulate
# noinspection PyUnresolvedReferences
from pprint import pprint
# noinspection PyUnresolvedReferences
import IPython

from nemere.utils.loader import SpecimenLoader
from fieldhunter.utils.base import Flows, NgramIterator, entropyFilteredOffsets, iterateSelected, intsFromNgrams, \
    ngramIsOverlapping
from fieldhunter.inference.fieldtypes import MSGtype


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


if __name__ == '__main__':
    parser = ArgumentParser(
        description='Re-Implementation of FieldHunter.')
    parser.add_argument('pcapfilename', help='Filename of the PCAP to load.')
    parser.add_argument('-i', '--interactive', help='open ipython prompt after finishing the analysis.',
                        action="store_true")
    args = parser.parse_args()

    if not isfile(args.pcapfilename):
        print('File not found: ' + args.pcapfilename)
        exit(1)
    pcapbasename = basename(args.pcapfilename)
    trace = splitext(pcapbasename)[0]
    # reportFolder = join(reportFolder, trace)
    # if not exists(reportFolder):
    #    makedirs(reportFolder)

    specimens = SpecimenLoader(args.pcapfilename)
    messages = list(specimens.messagePool.keys())
    flows = Flows(messages)

    msgtypefields = MSGtype(flows)
    # TODO The entropyThresh is not given in FH, so generate some statisics, illustrations,
    #   CDF, histograms, ... using our traces
    # print(tabulate(zip(msgtypefields.c2sEntropy, msgtypefields.s2cEntropy), headers=["c2s", "s2c"], showindex=True))


    # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
    # # # Working area
    # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #

    # MSG-Len (FH, Section 3.2.2)
    # application message length, linearly correlates with message size

    # coefficient threshold 0.6 (FH, Section 3.2.2)
    minCorrelation = 0.6
    # MSG-Len hypothesis threshold 0.9 (FH, Section 3.2.2)
    lenhypoThresh = 0.9

    acceptedCandidatesPerDir = list()  # type: List[Dict[int, int]]
    c2s, s2c = flows.splitDirections()  # type: List[L4NetworkMessage], List[L4NetworkMessage]
    for direction in [c2s, s2c]:  # per direction - for MSG-Len this is pointless, but the paper says to do it.
        # "stratifying messages by length": extract different size collection -> vector of message lengths
        keyfunc = lambda m: len(m.data)
        msgbylen = {k: list(v) for k, v in groupby(sorted(direction, key=keyfunc), keyfunc)}  # Homogeneous Size Collections
        minCollSize = min(len(v) for v in msgbylen.values())
        # generate size-heterogeneous collection by random sampling
        msgmixlen = list()
        for k, v in msgbylen.items():
            random.seed(42)
            if len(v) > minCollSize:
                msgmixlen.extend(random.sample(v, k=minCollSize))
            else:
                msgmixlen.extend(v)
        lens4msgmix = [len(m.data) for m in msgmixlen]

        candidateAtNgram = dict()
        # iterate n-grams' n=32, 24, 16 bits (4, 3, 2 bytes)
        for n in [4, 3, 2]:
            # entropy filter for each n-gram offset -> field values matrix
            offsets = entropyFilteredOffsets(msgmixlen, n)
            # TODO currently only big endian, see #intsFromNgrams
            ngIters = (intsFromNgrams(iterateSelected(NgramIterator(msg, n), offsets)) for msg in msgmixlen)
            ngramsAtOffsets = numpy.array(list(ngIters))  # TODO check if null byte ngrams cause problems

            # correlate columns of ngramsAtOffsets to lens4msgmix
            pearsonAtOffset = list()
            for ngrams in ngramsAtOffsets.T:
                # Pearson correlation coefficient (numeric value of n-gram) -> (len(msg.data))
                pearsonAtOffset.append(pearsonr(ngrams, lens4msgmix)[0])
            candidateAtNgram[n] = [o for pao, o in zip(pearsonAtOffset, offsets) if pao > minCorrelation]

        # verify length-hypothesis for candidates, solve for values at ngrams in candidateAtNgram (precedence for larger n)
        #   MSG_len = a * value + b (a > 0, b \in N)  - "Msg. Len. Model Parameters"
        #       lens4msgmix = ngramsAtOffsets[:,candidateAtNgram[n]] * a + 1 * b
        #   threshold 0.9 of the message pairs with different lengths
        acceptedCandidates = dict()  # type: Dict[int, int]
        acceptedX = dict()
        #           specifying found acceptable solutions at offset (key) with n (value) for this direction
        for n in [4, 3, 2]:
            for offset in candidateAtNgram[n]:
                # check precedence: if longer already-accepted n-gram overlaps this offset ignore
                if not checkPrecedence(offset, n, acceptedCandidates.items()):
                    continue
                # MSG-len hypothesis test - for ALL message pairs with different lengths (FH, 3.2.2 last paragraph)
                #   - for the n-grams from this offset - keep only those offsets, where the threshold of pairs holds
                solutionAcceptable = dict()  # type: Dict[Tuple[AbstractMessage, AbstractMessage], True]
                Xes = list()
                for l1, l2 in combinations(msgbylen.keys(),2):
                    for msg0, msg1 in product(msgbylen[l1], msgbylen[l2]):
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
                if acceptCount[True]/len(acceptCount) > lenhypoThresh:
                    acceptedCandidates[offset] = n
                    acceptedX[offset] = Xes
                    # TODO FH does not require this, but should not the values in A and B be equal within so that
                    #  a and b is one scalar value each?
        acceptedCandidatesPerDir.append(acceptedCandidates)
    print(tabulate(chain.from_iterable(acpd.items() for acpd in acceptedCandidatesPerDir)))
    # TODO create segments for each accepted candidate



    # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
    # TODO for validation, sub-class nemere.validation.messageParser.ParsingConstants263
    #   set TYPELOOKUP[x] to the value MSGtype.typelabel ("MSG-Type") for all fields in
    #   nemere.validation.messageParser.MessageTypeIdentifiers

    # interactive
    IPython.embed()

