"""
Only implements FH's binary message handling using n-grams (not textual using delimiters!)
"""

from argparse import ArgumentParser
from time import time, strftime
from openpyxl import Workbook
from os.path import join, exists

# noinspection PyUnresolvedReferences
from tabulate import tabulate
# noinspection PyUnresolvedReferences
from pprint import pprint
# noinspection PyUnresolvedReferences
import IPython

from nemere.utils.loader import SpecimenLoader
from nemere.utils.evaluationHelpers import StartupFilecheck, reportFolder
from nemere.utils.reportWriter import writeReport
from nemere.validation.dissectorMatcher import MessageComparator, DissectorMatcher

from fieldhunter.inference.fieldtypes import *
from fieldhunter.inference.common import segmentedMessagesAndSymbols
from fieldhunter.utils.base import Flows
from fieldhunter.utils.eval import FieldTypeReport




if __name__ == '__main__':
    parser = ArgumentParser(
        description='Re-Implementation of FieldHunter.')
    parser.add_argument('pcapfilename', help='Filename of the PCAP to load.')
    parser.add_argument('-i', '--interactive', help='open ipython prompt after finishing the analysis.',
                        action="store_true")
    # TODO remove these options: FH requires TCP/UDP over IP (FH, Section 6.6)
    # parser.add_argument('-l', '--layer', type=int, default=2,
    #                     help='Protocol layer relative to IP to consider. Default is 2 layers above IP '
    #                          '(typically the payload of a transport protocol).')
    # parser.add_argument('-r', '--relativeToIP', default=True, action='store_false')
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
    # print(tabulate(zip(msgtypefields.c2sEntropy, msgtypefields.s2cEntropy), headers=["c2s", "s2c"], showindex=True))

    # MSG-len
    print("Inferring", MSGlen.typelabel)
    msglenfields = MSGlen(flows)
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

    # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
    # # # TODO Working area
    # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #

    # FTR validation: calculate TP/FP/FN ==> P/R per protocol and per type
    ovTitle = "Overview"
    infieldWorkbook = Workbook()
    infieldWorkbook.active.title = ovTitle
    ovSheet = infieldWorkbook[ovTitle]
    ovSheet.append(FieldTypeReport.overviewHeaders)
    for infields in sortedInferredTypes:
        infieldReport = FieldTypeReport(infields, comparator)
        infieldReport.addXLworksheet(infieldWorkbook, ovTitle)
    infieldFilename = join(reportFolder,
                           f"FieldTypeReport_{filechecker.pcapstrippedname}_{strftime('%Y%m%d-%H%M%S')}.xlsx")
    if not exists(infieldFilename):
        print("Write field type report to", infieldFilename)
        infieldWorkbook.save(infieldFilename)
    else:
        print("Could not write", infieldFilename, "- File exists")
        for worksheet in infieldWorkbook.worksheets:
            headers = worksheet.rows[0]
            cells = worksheet.rows[1:]
            print( f"\nReport for {worksheet.title}:\n" + tabulate(cells, headers=headers) )



    # for later
    #
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

