"""
This script provides statistics about the given PCAP trace that have impact on the FieldHunter inference.
"""
# noinspection PyUnresolvedReferences
import IPython, logging
# noinspection PyUnresolvedReferences
from tabulate import tabulate
from argparse import ArgumentParser
from os.path import join
import matplotlib.pyplot as plt

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

    # # # # # # # # # # # # # # # # # #
    # Entropy plots: Relevant for MSG-Type and Trans-ID
    c2s, s2c = flows.splitDirections()
    c2sEntropy = pyitNgramEntropy(c2s, 1)
    s2cEntropy = pyitNgramEntropy(s2c, 1)
    fig: plt.Figure
    ax1: plt.Axes
    fig, (ax1, ax2) = plt.subplots(2,1,figsize=(6,6))
    for ax, entropy in [(ax1, c2sEntropy), (ax2, s2cEntropy)]:
        if len(entropy) > 0:
            ax.stem(entropy, use_line_collection=True)
        else:
            ax.text(1, .5, "no entries")
            ax.set_xlim(0, 32)
        ax.set_ylim(0.,1.)
        ax.grid(which="major", axis="y")
        ax.set_xlabel("byte offset")
        ax.set_ylabel("normalized entropy")
    plt.suptitle("Entropies per byte offset", fontsize="x-large")
    ax1.set_title("Client to Server Collection")
    ax2.set_title("Server to Client Collection")
    fig.tight_layout(rect=[0,0,1,.95])
    fig.savefig(join(reportFolder, filechecker.pcapstrippedname + ".pdf"))
    # # # # # # # # # # # # # # # # # #

    # # # # # # # # # # # # # # # # # #
    # DHCP "Transaction ID" that is a FH Session-ID
    if "dhcp" in specimens.pcapFileName:
        sessIDtuples = sorted( (
            (comparator.parsedMessages[specimens.messagePool[msg]].getValuesByName('dhcp.id')[0],
            msg.source.rpartition(':')[0], msg.destination.rpartition(':')[0]) for msg in messages),
            key = lambda x: x[0] )
        participantsTuples = [(a, *sorted([b, c])) for a, b, c in sessIDtuples]
        field2value = [(
            intsFromNgrams([bytes.fromhex(a)])[0],
            intsFromNgrams([bytes(map(int, b.split(".") + c.split(".")))])[0])
            for a, b, c in participantsTuples]
        ngSc = numpy.array(list(zip(*field2value)))
        catCorr = drv.information_mutual(ngSc[0], ngSc[1]) / drv.entropy_joint(ngSc)
        print(catCorr)
        # 0.5073953157493724
        # For dhcp_SMIA2011101X_deduped-1000.pcap this is just about .5 which is quite surprising.
        ignoreList = {"0.0.0.0", "255.255.255.255"}
        field2value = [(
            intsFromNgrams([bytes.fromhex(a)])[0],
            intsFromNgrams([bytes(map(int, b.split(".") + c.split(".")))])[0])
            for a, b, c in participantsTuples if b not in ignoreList and c not in ignoreList and a != "00000000"]
        ngSc = numpy.array(list(zip(*field2value)))
        catCorr = drv.information_mutual(ngSc[0], ngSc[1]) / drv.entropy_joint(ngSc)
        print(catCorr)
        # 0.566225418688138
        # Ignoring some trivial cases raises the correlation only marginally.
    # # # # # # # # # # # # # # # # # #

    # interactive
    if args.interactive:
        print()
        IPython.embed()
