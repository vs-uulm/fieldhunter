"""
This script provides statistics about the given PCAP trace that have impact on the FieldHunter inference.
"""
from argparse import ArgumentParser

import logging
from tabulate import tabulate
import IPython

from nemere.utils.loader import SpecimenLoader
from nemere.utils.evaluationHelpers import StartupFilecheck, reportFolder
from nemere.validation.dissectorMatcher import MessageComparator

from fieldhunter.inference.fieldtypes import *
from fieldhunter.utils.base import Flows
from fieldhunter.utils.eval import GroundTruth, csvAppend

logging.basicConfig(level=logging.DEBUG)
logging.getLogger().setLevel(logging.DEBUG)

if __name__ == '__main__':
    parser = ArgumentParser(
        description='Statistics about the given PCAP trace that have impact on the FieldHunter inference.')
    parser.add_argument('pcapfilename', help='Filename of the PCAP to load.')
    parser.add_argument('-i', '--interactive', help='open IPython prompt after finishing the analysis.',
                        action="store_true")
    args = parser.parse_args()

    filechecker = StartupFilecheck(args.pcapfilename)

    # FH always requires the protocol to be inside TCP/UDP over IP (FH, Section 6.6)
    specimens = SpecimenLoader(args.pcapfilename, layer=2, relativeToIP=True)
    # noinspection PyTypeChecker
    messages = list(specimens.messagePool.keys())  # type: List[L4NetworkMessage]
    comparator = MessageComparator(specimens, layer=2, relativeToIP=True)

    # # # # # # # # # # # # # # # # # #
    # Relevant for MSG-Type
    flows = Flows(messages)
    # print(tabulate(flows.c2sInConversations().keys()))
    # print(tabulate(flows.s2cInConversations().keys()))
    print("Conversations:\n")
    print(tabulate(flows.conversations().keys()))
    mqr = flows.matchQueryResponse()
    print("\nNumber of matching queries and responses:", len(mqr), "in", len(flows.flows), "flows")
    print("Found in", len(messages), f"messages. Coverage: {(len(mqr)*200)/len(messages):.1f}%")
    header = ["trace", "matching", "conversations", "flows", "messages", "coverage"]
    # amount/percentage of messages in the trace that are of "singular flows", i. e., a single message without either
    # a matching request or reply, is calculated by (100% - coverage).
    csvAppend(reportFolder, "flows", header, [[
        filechecker.pcapstrippedname, len(mqr), len(flows.conversations()), len(flows.flows),
        len(messages), (len(mqr)*200)/len(messages) ]])
    # TODO
    #   discern types: broadcasts, c2s/s2c without matching flow

    # # # # # # # # # # # # # # # # # #
    # Entropy filter threshold rationale: entropy statistics for ground truth fields
    # since the entropyThresh used in MSGtype/MSGlen (NonConstantNonRandomEntropyFieldType) is not given in FH
    # using our traces to back our value.
    gt = GroundTruth(comparator)
    gtTypeAndLengthEntropies = gt.typeAndLenEntropies()
    header = ["trace", "field name", "type label", "sample count", "entropy"]
    # write/append to a file. Columns: trace, field name, type label, sample count, entropy
    csvAppend(reportFolder, "typeAndLengthEntropies", header,
              ([filechecker.pcapstrippedname, *row] for row in gtTypeAndLengthEntropies if not numpy.isnan(row[-1])))
    # # # # # # # # # # # # # # # # # #


    # # # # # # # # # # # # # # # # # #
    # Relevant for MSG-Len
    # TODO length of messages, something like:
    #         keyfunc = lambda m: len(m.data)
    #         msgbylen = {k: v for k, v in groupby(sorted(direction, key=keyfunc), keyfunc)}
    # # # # # # # # # # # # # # # # # #


    # interactive
    if args.interactive:
        print()
        IPython.embed()
