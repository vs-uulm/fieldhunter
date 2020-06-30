from argparse import ArgumentParser
from os.path import isfile, basename, splitext

from tabulate import tabulate
from pprint import pprint
import IPython

from nemere.utils.loader import SpecimenLoader
from fieldhunter.utils.base import Flows, pyitEntropyFilterVertical


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

    # c2sEntropy = entropyFilterVertical(c2s)
    # s2cEntropy = entropyFilterVertical(s2c)
    # print(tabulate(zip(c2sEntropy, s2cEntropy), headers=["c2s", "s2c"], showindex=True))

    c2sEntropy = pyitEntropyFilterVertical(c2s)
    s2cEntropy = pyitEntropyFilterVertical(s2c)
    # print(tabulate(zip(c2sEntropy, s2cEntropy), headers=["c2s", "s2c"], showindex=True))

    mqr = flows.matchQueryRespone()

    # computer Q->R association/
    # Mutual information
    from pyitlib import discrete_random_variable as drv
    drv.information_mutual(ngramsfromQ, ngramsfromR)




    # Merge n-grams above causality threshold 0.8 and check correlation


    IPython.embed()

