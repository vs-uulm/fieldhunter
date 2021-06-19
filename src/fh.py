"""
Only implements FH's binary message handling using n-grams (not textual using delimiters!)
"""

from argparse import ArgumentParser
from time import time
from os.path import isfile, basename, splitext

# noinspection PyUnresolvedReferences
from netzob.Model.Vocabulary.Domain.Parser.MessageParser import InvalidParsingPathException
# noinspection PyUnresolvedReferences
from pprint import pprint
# noinspection PyUnresolvedReferences
import IPython

from nemere.utils.loader import SpecimenLoader
from nemere.validation.dissectorMatcher import MessageComparator, DissectorMatcher
from nemere.utils.reportWriter import writeReport
from fieldhunter.inference.fieldtypes import *
from fieldhunter.inference.common import segmentedMessagesAndSymbols
from fieldhunter.utils.base import Flows


if __name__ == '__main__':
    parser = ArgumentParser(
        description='Re-Implementation of FieldHunter.')
    parser.add_argument('pcapfilename', help='Filename of the PCAP to load.')
    parser.add_argument('-i', '--interactive', help='open ipython prompt after finishing the analysis.',
                        action="store_true")
    parser.add_argument('-l', '--layer', type=int, default=2,
                        help='Protocol layer relative to IP to consider. Default is 2 layers above IP '
                             '(typically the payload of a transport protocol).')
    parser.add_argument('-r', '--relativeToIP', default=False, action='store_true')
    args = parser.parse_args()

    if not isfile(args.pcapfilename):
        print('File not found: ' + args.pcapfilename)
        exit(1)
    pcapbasename = basename(args.pcapfilename)
    trace = splitext(pcapbasename)[0]
    # reportFolder = join(reportFolder, trace)
    # if not exists(reportFolder):
    #    makedirs(reportFolder)

    specimens = SpecimenLoader(args.pcapfilename, layer=args.layer, relativeToIP = args.relativeToIP)
    # noinspection PyTypeChecker
    messages = list(specimens.messagePool.keys())  # type: List[L2NetworkMessage]
    flows = Flows(messages)

    # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
    print("Hunting fields in", trace)
    inferenceStart = time()

    # MSG-type
    print("Inferring", MSGtype.typelabel)
    msgtypefields = MSGtype(flows)
    # TODO The entropyThresh is not given in FH, so generate some statisics, illustrations,
    #   CDF, histograms, ... using our traces
    # print(tabulate(zip(msgtypefields.c2sEntropy, msgtypefields.s2cEntropy), headers=["c2s", "s2c"], showindex=True))

    # MSG-len
    print("Inferring", MSGlen.typelabel)
    msglenfields = MSGlen(flows)
    # print(tabulate(list(msglenfields.acceptedCandidatesPerDir[0].items()) + ["--"]
    #                + list(msglenfields.acceptedCandidatesPerDir[1].items())))

    # Host-ID
    print("Inferring", HostID.typelabel)
    hostidfields = HostID(messages)
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

    # Session-ID (FH, Section 3.2.4)
    print("Inferring", SessionID.typelabel)
    sessionidfields = SessionID(messages)
    # # Problem similar to Host-ID leads to same bad performance.
    # # Moreover, Host-ID will always return a subset of Session-ID fields, so Host-ID should get precedence.
    # print(HostID.catCorrPosLen(sessionidfields.categoricalCorrelation))

    # Trans-ID (FH, Section 3.2.5)
    print("Inferring", TransID.typelabel)
    transidfields = TransID(flows)
    # pprint(transidfields.segments)

    # Accumulators (FH, Section 3.2.6)
    print("Inferring", Accumulator.typelabel)
    accumulatorfields = Accumulator(flows)
    # pprint(accumulatorfields.segments)

    segmentedMessages, symbols = segmentedMessagesAndSymbols(
        # in order of fieldtypes.precedence!
        (msgtypefields, msglenfields,
         # Host-ID will always return a subset of Session-ID fields, so Host-ID should get precedence
         hostidfields, sessionidfields,
         transidfields, accumulatorfields),
        messages
    )

    inferenceDuration = time() - inferenceStart
    # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #

    comparator = MessageComparator(specimens, layer=args.layer, relativeToIP=args.relativeToIP)
    comparator.pprintInterleaved(symbols)

    # calc FMS per message
    print("Calculate FMS...")
    message2quality = DissectorMatcher.symbolListFMS(comparator, symbols)

    # write statistics to csv
    writeReport(message2quality, inferenceDuration, specimens, comparator, "fieldhunter-literal")


    # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
    # # # TODO Working area
    # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #

    # TODO introduce doctests to check critical functions

    # TODO finalize a "literal" implementation: main script, "set aside" the fieldtypes module

    # TODO derive a "improved" implementation: main script, copy/subclass the fieldtypes module and address the todos there

    # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
    # TODO for FTR validation, sub-class nemere.validation.messageParser.ParsingConstantsXXX
    #   set TYPELOOKUP[x] to the value FieldType.typelabel (e. g., "MSG-Type") for all fields in
    #   nemere.validation.messageParser.MessageTypeIdentifiers

    # TODO FTR validation: compare to dissector/field type

    # interactive
    if args.interactive:
        # noinspection PyUnresolvedReferences
        from tabulate import tabulate
        IPython.embed()

