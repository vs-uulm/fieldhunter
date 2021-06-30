"""
Only implements FH's binary message handling using n-grams (not textual using delimiters!)
"""
import logging
from argparse import ArgumentParser
from time import time

# noinspection PyUnresolvedReferences
from tabulate import tabulate
# noinspection PyUnresolvedReferences
from pprint import pprint
# noinspection PyUnresolvedReferences
import IPython

from nemere.utils.loader import SpecimenLoader
from nemere.utils.evaluationHelpers import StartupFilecheck
from nemere.utils.reportWriter import writeReport
from nemere.validation.dissectorMatcher import MessageComparator, DissectorMatcher

from fieldhunter.inference.fieldtypesRelaxed import *
from fieldhunter.inference.common import segmentedMessagesAndSymbols
from fieldhunter.utils.base import Flows
from fieldhunter.utils.eval import FieldTypeReport


logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger()
logger.setLevel(logging.DEBUG)


if __name__ == '__main__':
    parser = ArgumentParser(
        description='Re-Implementation of FieldHunter.')
    parser.add_argument('pcapfilename', help='Filename of the PCAP to load.')
    parser.add_argument('-i', '--interactive', help='open ipython prompt after finishing the analysis.',
                        action="store_true")
    args = parser.parse_args()
    layer = 2
    relativeToIP = True

    filechecker = StartupFilecheck(args.pcapfilename)

    specimens = SpecimenLoader(args.pcapfilename, layer = layer, relativeToIP = relativeToIP)
    # noinspection PyTypeChecker
    messages = list(specimens.messagePool.keys())  # type: List[L4NetworkMessage]
    flows = Flows(messages)

    # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
    print("Hunting fields in", filechecker.pcapstrippedname)
    inferenceStart = time()

    # MSG-type
    print("Inferring", MSGtype.typelabel)
    msgtypefields = MSGtype(flows)

    # MSG-len
    print("Inferring", MSGlen.typelabel)
    msglenfields = MSGlen(flows)

    # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
    # # # TODO Working area
    # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #

    # print(tabulate(list(msglenfields.acceptedCandidatesPerDir[0].items()) + ["--"]
    #                + list(msglenfields.acceptedCandidatesPerDir[1].items())))

    # Host-ID
    print("Inferring", HostID.typelabel)
    hostidfields = HostID(messages)
    # print(HostID.catCorrPosLen(hostidfields.categoricalCorrelation))

    # Session-ID (FH, Section 3.2.4)
    print("Inferring", SessionID.typelabel)
    sessionidfields = SessionID(messages)
    # print(HostID.catCorrPosLen(sessionidfields.categoricalCorrelation))

    # Trans-ID (FH, Section 3.2.5)
    print("Inferring", TransID.typelabel)
    transidfields = TransID(flows)
    # pprint(transidfields.segments)

    # Accumulators (FH, Section 3.2.6)
    print("Inferring", Accumulator.typelabel)
    accumulatorfields = Accumulator(flows)
    # pprint(accumulatorfields.segments)

    # in order of fieldtypes.precedence!
    sortedInferredTypes = sorted(
        (msglenfields, msgtypefields, hostidfields, sessionidfields, transidfields, accumulatorfields),
        key=lambda l: precedence[l.typelabel] )
    segmentedMessages, symbols = segmentedMessagesAndSymbols(sortedInferredTypes, messages)

    inferenceDuration = time() - inferenceStart
    # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #

    print(tabulate(
        [(infield.typelabel,
            sum(1 for msgsegs in infield.segments if len(msgsegs) > 0),
            max(len(msgsegs) for msgsegs in infield.segments)
                if len(infield.segments) > 0 else 0 # prevent empty sequence for max()
        ) for infield in sortedInferredTypes],
        headers=["typelabel","messages","max inferred per msg"]
    ))

    # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #

    nontrivialSymbols = [sym for sym in symbols if len(sym.fields) > 1]
    comparator = MessageComparator(specimens, layer=layer, relativeToIP=relativeToIP)
    print("Dissection complete.")
    comparator.pprintInterleaved(nontrivialSymbols)
    print(f"\n   + {len(symbols)-len(nontrivialSymbols)} messages without any inferred fields.")

    # calc FMS per message
    print("Calculate FMS...")
    message2quality = DissectorMatcher.symbolListFMS(comparator, symbols)
    # write statistics to csv
    writeReport(message2quality, inferenceDuration, specimens, comparator, "fieldhunter-literal",
                filechecker.reportFullPath)

    # FTR validation: calculate TP/FP/FN ==> P/R per protocol and per type
    infieldWorkbook = FieldTypeReport.newWorkbook()
    for infields in sortedInferredTypes:
        infieldReport = FieldTypeReport(infields, comparator, segmentedMessages)
        infieldReport.addXLworksheet(infieldWorkbook, FieldTypeReport.ovTitle)
    FieldTypeReport.saveWorkbook(infieldWorkbook, filechecker.pcapstrippedname)


    # TODO derive an "improved" implementation:
    #  new separate main script,
    #  define a collection of base classes for the literal and improved implementations,
    #  copy/subclass the fieldtypes module and address the todos there

    # msglenfields
    # msgtypefields
    # hostidfields
    # sessionidfields
    # transidfields
    # accumulatorfields

    # interactive
    if args.interactive:
        IPython.embed()

