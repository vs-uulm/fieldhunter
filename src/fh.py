from argparse import ArgumentParser
from os.path import isfile, basename, splitext

from tabulate import tabulate
from pprint import pprint
import IPython

from nemere.utils.loader import SpecimenLoader
from fieldhunter.utils.base import NgramIterator, splitDirections, entropyFilterVertical


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

    c2s, s2c = splitDirections(messages)

    pprint(c2s)
    pprint(s2c)

    c2sEntropy = entropyFilterVertical(c2s)
    s2cEntropy = entropyFilterVertical(s2c)



    # m0n4 = NgramIterator(messages[0], 4)
    # m0n4list = list(m0n4)
    # print(tabulate(m0n4list))

    IPython.embed()

