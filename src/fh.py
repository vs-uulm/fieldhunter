from argparse import ArgumentParser
from os.path import isfile, basename, splitext

from tabulate import tabulate
from pprint import pprint
import IPython

from nemere.utils.loader import SpecimenLoader
from nemere.inference.segments import TypedSegment
from nemere.inference.analyzers import Value
from fieldhunter.utils.base import Flows, pyitEntropyVertical, qrAssociationCorrelation, verticalByteMerge, \
    mutualInformation, list2ranges

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

    c2s, s2c = flows.splitDirections()
    # pprint(c2s)
    # pprint(s2c)

    # c2sEntropy = entropyVertical(c2s)
    # s2cEntropy = entropyVertical(s2c)
    # print(tabulate(zip(c2sEntropy, s2cEntropy), headers=["c2s", "s2c"], showindex=True))

    # discard constant and random offsets (threshold?)
    entropyThresh = 0.4  # Not given in FH!
    c2sEntropy = pyitEntropyVertical(c2s)
    c2sEntropyFiltered = [offset for offset, entropy in enumerate(c2sEntropy) if 0 < entropy < entropyThresh]
    s2cEntropy = pyitEntropyVertical(s2c)
    s2cEntropyFiltered = [offset for offset, entropy in enumerate(s2cEntropy) if 0 < entropy < entropyThresh]
    # print(tabulate(zip(c2sEntropy, s2cEntropy), headers=["c2s", "s2c"], showindex=True))
    # print(c2sEntropyFiltered)
    # print(s2cEntropyFiltered)

    from collections import Counter
    # print(tabulate(Counter(msg.data[2:4].hex() for msg in c2s).most_common()))
    # print(tabulate(Counter(msg.data[2:4].hex() for msg in s2c).most_common()))



    # compute Q->R association/
    mqr = flows.matchQueryRespone()
    # Mutual information
    qrCausality = qrAssociationCorrelation(mqr)
    # filter: only if offset is in c2sEntropyFiltered/s2cEntropyFiltered and the causality is greater than the causalityThresh
    causalityThresh = 0.6  # FH, Sec. 3.2.1 says 0.8
    filteredCausality = {offset: qrCausality[offset] for offset in
                         set(c2sEntropyFiltered).intersection(s2cEntropyFiltered)
                         if qrCausality[offset] > causalityThresh}
    # filteredCausality are offsets of MSG-Type candidate n-grams
    print(tabulate(sorted(filteredCausality.items())))


    # Merge n-grams above causality threshold and check correlation
    mergingOffsets = list()
    for offset in sorted(filteredCausality.keys()):
        mergingOffsets.append(offset)
        qMergedField, rMergedField = verticalByteMerge(mqr, mergingOffsets)
        mergedCausality = mutualInformation(qMergedField, rMergedField)
        if mergedCausality <= causalityThresh:
            # Filter problematic n-grams
            mergingOffsets.pop()
    # re-calculate in case the last iteration removed a problematic n-gram
    qMergedField, rMergedField = verticalByteMerge(mqr, mergingOffsets)
    mergedCausality = mutualInformation(qMergedField, rMergedField)

    print("mergedCausality", mergedCausality)
    print("mergingOffsets", mergingOffsets)
    print("  from offsets", sorted(filteredCausality.keys()))

    # create segments from bytes in mergingOffsets and compare to dissector/field type
    msgtypeRanges = list2ranges(mergingOffsets)
    msgtypeSegments = list()
    for message in c2s + s2c:
        segs4msg = list()
        for start,end in msgtypeRanges:
            segs4msg.append(TypedSegment(Value(message), start, end+1-start, "MSG-Type"))
        msgtypeSegments.append(segs4msg)


    # interactive
    IPython.embed()

