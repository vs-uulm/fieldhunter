"""
This script provides statistics about the given PCAP trace that have impact on the FieldHunter inference.
"""
from argparse import ArgumentParser
from os.path import isfile, basename, splitext
from tabulate import tabulate

from nemere.utils.loader import SpecimenLoader
from fieldhunter.inference.fieldtypes import *
from fieldhunter.utils.base import Flows


if __name__ == '__main__':
    parser = ArgumentParser(
        description='Statistics about the given PCAP trace that have impact on the FieldHunter inference.')
    parser.add_argument('pcapfilename', help='Filename of the PCAP to load.')
    parser.add_argument('-i', '--interactive', help='open IPython prompt after finishing the analysis.',
                        action="store_true")
    args = parser.parse_args()

    if not isfile(args.pcapfilename):
        print('File not found: ' + args.pcapfilename)
        exit(1)
    pcapbasename = basename(args.pcapfilename)
    trace = splitext(pcapbasename)[0]

    # FH always requires the protocol to be inside TCP/UDP over IP (FH, Section 6.6)
    specimens = SpecimenLoader(args.pcapfilename, layer=2, relativeToIP=True)
    # noinspection PyTypeChecker
    messages = list(specimens.messagePool.keys())  # type: List[L4NetworkMessage]

    # # # # # # # # # # # # # # # # # #
    # Relevant for MSG-Type
    flows = Flows(messages)
    print(tabulate(flows.c2sInConversations().keys()))
    print(tabulate(flows.s2cInConversations().keys()))
    print(tabulate(flows.conversations().keys()))
    mqr = flows.matchQueryResponse()
    print("Number of matches queries and responses:", len(mqr), "in", len(flows.flows), "flows")
    print("Found in", len(messages), "messages. Coverage:", (len(mqr)*200)/len(messages), "%")
    #
    # TODO amount/percentage of messages in the trace that are of singular flows,
    #   i. e. without a matching request or reply
    #   discern types: broadcasts, c2s/s2c without matching flow
    #
    # Entropy filter threshold rationale -> e. g. some histogram, CDF, ...
    # # # # # # # # # # # # # # # # # #


    # # # # # # # # # # # # # # # # # #
    # Relevant for MSG-Len
    # TODO length of messages, something like:
    #         keyfunc = lambda m: len(m.data)
    #         msgbylen = {k: v for k, v in groupby(sorted(direction, key=keyfunc), keyfunc)}
    # # # # # # # # # # # # # # # # # #
