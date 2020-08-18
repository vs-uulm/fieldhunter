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
from fieldhunter.inference.fieldtypes import *
from nemere.inference.analyzers import Value
from nemere.inference.segments import TypedSegment
from fieldhunter.utils.base import Flows, list2ranges
from fieldhunter.utils.base import NgramIterator, iterateSelected, intsFromNgrams, ngramIsOverlapping


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
    messages = list(specimens.messagePool.keys())  # type: List[L2NetworkMessage]
    flows = Flows(messages)

    # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
    # TODO reactivate finally

    # # MSG-type
    # msgtypefields = MSGtype(flows)
    # TODO The entropyThresh is not given in FH, so generate some statisics, illustrations,
    #   CDF, histograms, ... using our traces
    # print(tabulate(zip(msgtypefields.c2sEntropy, msgtypefields.s2cEntropy), headers=["c2s", "s2c"], showindex=True))

    # # MSG-len
    # msglenfields = MSGlen(flows)
    # print(tabulate(list(msglenfields.acceptedCandidatesPerDir[0].items()) + ["--"]
    #                + list(msglenfields.acceptedCandidatesPerDir[1].items())))

    # # Host-ID
    # hostidfields = HostID(messages)
    # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
    # # # # Investigate low categoricalCorrelation for all but one byte within an address field (see NTP and DHCP).
    # # # # According to NTP offset 12 (REF ID, often DST IP address) and DHCP offsets (12, 17, and) 20 (IPs)
    # # # # this works in principle, but if the n-gram is too short the correlation gets lost some n-grams.
    # # print(tabulate(zip(*[categoricalCorrelation]), showindex="always"))
    # # from matplotlib import pyplot
    # # pyplot.bar(range(len(categoricalCorrelation)), categoricalCorrelation)
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
    # # # #  Probably it does not mention a Entropy filter, direction separation, or - most probably -
    # # # #  an iterative n-gram size
    # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
    # print(HostID.catCorrPosLen(hostidfields.categoricalCorrelation))

    # # Session-ID (FH, Section 3.2.4)
    # sessionidfields = SessionID(messages)
    # # Problem similar to Host-ID leads to same bad performance.
    # # Moreover, Host-ID will always return a subset of Session-ID fields, so Host-ID should get precedence.
    # print(HostID.catCorrPosLen(sessionidfields.categoricalCorrelation))

    # # Trans-ID
    # transidfields = TransID(flows)
    # pprint(transidfields.segments)

    pass
    # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
    # # # TODO Working area
    # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #

    # TODO Accumulators (FH, Section 3.2.6)
    #   "Accumulators are fields that have increasing values over consecutive message within the same conversation." (FH, Sec. 3.2.6)

    endianness = 'big'
    ns = (8, 4, 3, 2)
    deltaEntropyThresh = 0.8  # Not given in FH, own empirics: 0.2

    # c2s and s2c independently
    c2sConvs = {key: list(sorted(conv, key=lambda m: m.date)) for key,conv in flows.c2sInConversations().items()}
    s2cConvs = {key: list(sorted(conv, key=lambda m: m.date)) for key,conv in flows.s2cInConversations().items()}
    c2sdeltas = None
    s2cdeltas = None
    c2sDeltaEntropies = dict()
    s2cDeltaEntropies = dict()
    for direction in (c2sConvs, s2cConvs):
        # deltas per offset and n over all message-pairs of all conversations
        deltas = dict()
        for key, conv in direction.items():
            if len(conv) > 2:
                continue
            # subsequent messages per direction per conversation
            for msgA, msgB in zip(conv[:-1], conv[1:]):
                # iterate n-grams' n = 8, 4, 3, 2
                # combined from Sec. 3.1.2: n=32, 24, 16 bits (4, 3, 2 bytes)
                #       and see Sec. 3.2.6: n=64, 32, 16 bits (8, 4, 2 bytes)
                for n in ns:
                    if n not in deltas:
                        deltas[n] = dict()
                    for offset, (ngramA, ngramB) in enumerate(zip(NgramIterator(msgA, n), NgramIterator(msgB, n))):
                        # calculate delta between n-grams (n and offset identical) two subsequent messages
                        # TODO also support little endian
                        delta = int.from_bytes(ngramB, endianness) - int.from_bytes(ngramA, endianness)
                        # "compress": ln delta
                        if offset not in deltas[n]:
                            deltas[n][offset] = list()
                        deltas[n][offset].append(delta)  # log(
        # check: delta positive and fairly constant
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

        if direction == c2sConvs:
            c2sdeltas = deltas
            c2sDeltaEntropies = deltaEntropies
        if direction == s2cConvs:
            s2cdeltas = deltas
            s2cDeltaEntropies = deltaEntropies

    print('c2sDeltaEntropies (n: offset: value)')
    pprint(c2sDeltaEntropies)
    # print('s2cDeltaEntropies (n: offset: value)')
    # pprint(s2cDeltaEntropies)

    # "fairly constant": relatively low entropy -> threshold (value not given in FH)
    # TODO add s2cDeltaEntropies
    c2sFilteredDE = {n: {offs: entr for offs, entr in offsdelt.items() if entr < deltaEntropyThresh}
                     for n, offsdelt in c2sDeltaEntropies.items()}
    candidates = dict()  # type: Dict[int, List[int]]
    for n in reversed(sorted(c2sFilteredDE.keys())):
        # no offsets for ngram size
        if len(c2sFilteredDE[n]) == 0:
            continue
        for offset in sorted(c2sFilteredDE[n].keys()):
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

    # create segments from candidates
    posLen = [(o, n) for n, offsets in candidates.items() for o in offsets]

    

    # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
    # TODO for validation, sub-class nemere.validation.messageParser.ParsingConstantsXXX
    #   set TYPELOOKUP[x] to the value FieldType.typelabel (e. g., "MSG-Type") for all fields in
    #   nemere.validation.messageParser.MessageTypeIdentifiers

    # interactive
    IPython.embed()

