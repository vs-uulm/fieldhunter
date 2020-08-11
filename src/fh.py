"""
Only implements FH's binary message handling using n-grams (not textual using delimiters!)
"""

from argparse import ArgumentParser
from os.path import isfile, basename, splitext

from typing import Dict, Tuple, Iterable, Sequence, List
from itertools import groupby, product, chain, combinations
from collections import Counter
import random, numpy
from scipy.stats import pearsonr
from netzob.Model.Vocabulary.Messages.AbstractMessage import AbstractMessage

# noinspection PyUnresolvedReferences
from tabulate import tabulate
# noinspection PyUnresolvedReferences
from pprint import pprint
# noinspection PyUnresolvedReferences
import IPython

from nemere.utils.loader import SpecimenLoader
from nemere.inference.analyzers import Value
from nemere.inference.segments import TypedSegment
from fieldhunter.utils.base import Flows, list2ranges
from fieldhunter.inference.fieldtypes import MSGtype, MSGlen
from fieldhunter.utils.base import NgramIterator, entropyFilteredOffsets, iterateSelected, intsFromNgrams, ngramIsOverlapping


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

    # TODO reactivate finally

    # msgtypefields = MSGtype(flows)
    # TODO The entropyThresh is not given in FH, so generate some statisics, illustrations,
    #   CDF, histograms, ... using our traces
    # print(tabulate(zip(msgtypefields.c2sEntropy, msgtypefields.s2cEntropy), headers=["c2s", "s2c"], showindex=True))

    # msglenfields = MSGlen(flows)
    # print(tabulate(list(msglenfields.acceptedCandidatesPerDir[0].items()) + ["--"]
    #                + list(msglenfields.acceptedCandidatesPerDir[1].items())))

    # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
    # # # TODO Working area
    # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #

    # TODO Host-ID
    # Host identifier (Host-ID) inference (FH, Sec. 3.2.3)
    # Find n-gram that is strongly correlated with IP address of sender

    from pyitlib import discrete_random_variable as drv

    typelabel = 'Host-ID'

    hostCorrelationThresh = 0.9  # 0.9, threshold for correlation between host ID and IP address (FH, Sec. 3.2.3)
    minHostLenThresh = 4  # host ID fields must at least be 4 bytes long (FH, Sec. 3.2.3)

    # ngram at offset and src address
    ngramsSrcs = list()
    categoricalCorrelation = list()
    # recover byte representation of ipv4 address from Netzob message and make one int out if each
    srcs = intsFromNgrams([bytes(map(int, msg.source.rpartition(':')[0].split('.'))) for msg in messages])
    # Host-ID uses 8-bit/1-byte n-grams according to FH, Sec. 3.1.2, but this does not work well (see below)
    for ngrams in zip(*(NgramIterator(msg, n=1) for msg in messages)):
        ngSc = numpy.array([intsFromNgrams(ngrams), srcs])
        # categoricalCorrelation: R(x, y) = I(x: y)/H(x, y) \in [0,1]
        catCorr = drv.information_mutual(ngSc[0], ngSc[1]) / drv.entropy_joint(ngSc)
        ngramsSrcs.append(ngSc)
        categoricalCorrelation.append(catCorr)

    # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
    # # # Investigate low categoricalCorrelation for all but one byte within an address field (see NTP and DHCP).
    # # # According to NTP offset 12 (REF ID, often DST IP address) and DHCP offsets (12, 17, and) 20 (IPs)
    # # # this works in principle, but if the n-gram is too short the correlation gets lost some n-grams.
    # print(tabulate(zip(*[categoricalCorrelation]), showindex="always"))
    # from matplotlib import pyplot
    # pyplot.bar(range(len(categoricalCorrelation)), categoricalCorrelation)
    # pyplot.show()
    # # sum([msg.data[20:24] == bytes(map(int, msg.source.rpartition(':')[0].split('.'))) for msg in messages])
    # # sum([int.from_bytes(messages[m].data[20:24], "big") == srcs[m] for m in range(len(messages))])
    # # # While the whole bootp.ip.server [20:24] correlates nicely to the IP address, single n-grams don't.
    # serverIP = [(int.from_bytes(messages[m].data[20:24], "big"), srcs[m]) for m in range(len(messages))]
    # serverIP0 = [(messages[m].data[20], srcs[m]) for m in range(len(messages))]
    # serverIP1 = [(messages[m].data[21], srcs[m]) for m in range(len(messages))]
    # serverIP2 = [(messages[m].data[22], srcs[m]) for m in range(len(messages))]
    # serverIP3 = [(messages[m].data[23], srcs[m]) for m in range(len(messages))]
    # # nsp = numpy.array([sip for sip in serverIP])
    # # # The correlation is perfect, if null values are omitted
    # nsp = numpy.array([sip for sip in serverIP if sip[0] != 0])   #  and sip[0] == sip[1]
    # # nsp0 = numpy.array(serverIP0)
    # # nsp1 = numpy.array(serverIP1)
    # # nsp2 = numpy.array(serverIP2)
    # # nsp3 = numpy.array(serverIP3)
    # nsp0 = numpy.array([sip for sip in serverIP0 if sip[0] != 0])
    # nsp1 = numpy.array([sip for sip in serverIP1 if sip[0] != 0])
    # nsp2 = numpy.array([sip for sip in serverIP2 if sip[0] != 0])
    # nsp3 = numpy.array([sip for sip in serverIP3 if sip[0] != 0])
    # for serverSrcPairs in [nsp, nsp0, nsp1, nsp2, nsp3]:
    #     print(drv.information_mutual(serverSrcPairs[:, 0], serverSrcPairs[:, 1]) / drv.entropy_joint(serverSrcPairs.T))
    # # # This is no implementation error, but raises doubts about the Host-ID description completeness:
    # # # Probably it does not mention a Entropy filter, direction separation, or - most probably - an iterative n-gram size
    # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #

    # merge consecutive candidate n-grams with categoricalCorrelation > hostCorrelationThresh
    catCorrOffsets = [ offset for offset, catCorr in enumerate(categoricalCorrelation)
                       if catCorr > hostCorrelationThresh ]
    catCorrRanges = list2ranges(catCorrOffsets)

    # discard short fields < minHostLenThresh
    catCorrPosLen = [ (start, end+1-start) for start, end in catCorrRanges if end+1-start >= minHostLenThresh ]

    # Generate Segments from remaining field ranges
    segments = list()
    for message in messages:
        mval = Value(message)
        segs4msg = list()
        for start, length in catCorrPosLen:
            segs4msg.append(TypedSegment(mval, start, length, typelabel))
        segments.append(segs4msg)

    # TODO move to class


    # TODO Session-ID (FH, Section 3.2.4)
    # most of FH, Section 3.2.4 refers to Host-ID, so we use all missing details from there and reuse the implementation
    # get srcs (see Host-ID) and dst in same manner
    # iterate the ngrams (n=1) and create a ngScDs (instead of just ngSc: ngram/source/destination)
    # correlate n-grams to (client IP, server IP) tuple
    #  by calculating the catCorr for the ngram and the source/destination tuple (check out how to nest the tuple right)
    # categorical correlation like Host-ID
    # create a common base class providing reusable code

    # TODO Trans-ID (FH, Section 3.2.5, Fig. 3 right)
    # TODO Accumulators (FH, Section 3.2.6)
    # iterate n-grams' n=32, 24, 16 bits (4, 3, 2 bytes), see 3.1.2


    # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
    # TODO for validation, sub-class nemere.validation.messageParser.ParsingConstants263
    #   set TYPELOOKUP[x] to the value MSGtype.typelabel ("MSG-Type") for all fields in
    #   nemere.validation.messageParser.MessageTypeIdentifiers

    # interactive
    IPython.embed()

