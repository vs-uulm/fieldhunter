from argparse import ArgumentParser
from os.path import isfile, basename, splitext

from tabulate import tabulate
from pprint import pprint
import IPython

from nemere.utils.loader import SpecimenLoader
from fieldhunter.utils.base import Flows, pyitEntropyVertical, qrAssociation


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
    entropyThresh = 0.2
    c2sEntropy = pyitEntropyVertical(c2s)
    c2sEntropyFiltered = [offset for offset, entropy in enumerate(c2sEntropy) if 0 < entropy < entropyThresh]
    s2cEntropy = pyitEntropyVertical(s2c)
    s2cEntropyFiltered = [offset for offset, entropy in enumerate(s2cEntropy) if 0 < entropy < entropyThresh]
    # print(tabulate(zip(c2sEntropy, s2cEntropy), headers=["c2s", "s2c"], showindex=True))
    print(c2sEntropyFiltered)
    print(s2cEntropyFiltered)

    mqr = flows.matchQueryRespone()

    # compute Q->R association/
    # Mutual information
    qrA = qrAssociation(mqr)
    # consider only if c2sEntropyFiltered/s2cEntropyFiltered holds






    # Merge n-grams above causality threshold 0.8 and check correlation


    IPython.embed()

