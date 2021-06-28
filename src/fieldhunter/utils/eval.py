import os, csv, logging
from typing import Any
from collections import Counter

from nemere.validation.dissectorMatcher import MessageComparator

from fieldhunter.inference.fieldtypes import *


def csvAppend(reportFolder: str, fileName: str, header: List[str], rows: Iterable[Iterable[Any]]):
    csvpath = os.path.join(reportFolder, fileName + '.csv')
    csvWriteHead = False if os.path.exists(csvpath) else True

    print('Write statistics to {}...'.format(csvpath))
    with open(csvpath, 'a') as csvfile:
        statisticscsv = csv.writer(csvfile)
        if csvWriteHead:
            statisticscsv.writerow(header)
        statisticscsv.writerows(rows)


class FieldTypeReport(object):

    headers = ["hexbytes", "segment offset", "segment end",
               "overlap ratio", "overlap index", "overlap offset", "overlap end", "overlap value",
               "message date", "message type", "field name", "field type"]

    def __init__(self, fieldtype: FieldType, comparator: MessageComparator):
        self._fieldtype = fieldtype
        self._comparator = comparator

    def lookupOverlap(self):

        tabdata = list()

        for seg in (seg for msgsegs in self._fieldtype.segments for seg in msgsegs if msgsegs):
            # field: from ground true; seg(ment): inferred; overlap: intersection of field and segment
            overlapRatio, overlapIndex, overlapOffset, overlapEnd = self._comparator.fieldOverlap(seg)
            messagetype, fieldname, fieldtype = self._comparator.lookupField(seg)
            overlapValue = "'" + seg.message.data[overlapOffset:overlapEnd].hex() + "'"

            tabdata.append(["'" + seg.bytes.hex() + "'", seg.offset, seg.nextOffset,
                            overlapRatio, overlapIndex, overlapOffset, overlapEnd, overlapValue,
                            seg.message.date, messagetype, fieldname, fieldtype])

            # TODO determine what is a TP/FP using GroundTruth
            # TODO incorporate the precedence of multiple overlapping inferred fields (column: "hidden by other type")
            #

        return tabdata

    @property
    def typelabel(self):
        return self._fieldtype.typelabel

    # TODO write to file




class GroundTruth(object):
    """tshark dissector field names for sample protocols mapped from the FieldHunter field type class."""
    fieldtypes = {
        MSGlen.typelabel:    ["nbss.length"],
        MSGtype.typelabel:   ["dhcp.option.dhcp", "ntp.flags", "ntp.stratum", "dns.flags",
                              "nbns.flags", "smb.cmd", "smb.flags", ],
        HostID.typelabel:    ["dhcp.ip.client", "dhcp.ip.your", "dhcp.ip.server", "ntp.refid"],
        SessionID.typelabel: ["dhcp.id", "smb.pid", "smb.uid", "smb.mid"],
        TransID.typelabel:   ["dns.id"],
        Accumulator.typelabel: []
    }

    def __init__(self, comparator:MessageComparator, endianness: str = "big"):
        self._comparator = comparator
        self._endianness = endianness
        logging.getLogger(__name__).setLevel(logging.DEBUG)

    def entropyPerField(self, fieldname: str):
        """Collect true fields values and calculate their entropy for the current trace."""
        fieldsValues = [bytes.fromhex(hexval) for hexval in self._comparator.lookupValues4FieldName(fieldname)]
        if len(fieldsValues) > 0:
            fieldLengths = Counter(len(bv) for bv in fieldsValues)
            # should normally be a constant value for this kind of fields
            mostCommonLen = fieldLengths.most_common(1)[0][0]
            logging.getLogger(__name__).debug(f"Field lengths of {fieldname}: {repr(fieldLengths)}")
            entropy = drv.entropy(intsFromNgrams(fieldsValues, self._endianness)) / (mostCommonLen * 8)
        else:
            entropy = numpy.nan
        return len(fieldsValues), entropy

    def typeAndLenEntropies(self):
        """
        Collect MSGtype/MSGlen true fields according to GroundTruth.fieldtypes[MSGtype.typelabel/MSGlen.typelabel]

        :return: list of lists of "field name", "type label", "sample count", and "entropy"
        """
        entropyList = list()
        for typelabel in [MSGtype.typelabel, MSGlen.typelabel]:
            for fieldname in GroundTruth.fieldtypes[typelabel]:
                # for each field name calculate entropy
                entropyList.append([
                    fieldname,
                    typelabel,
                    *self.entropyPerField(fieldname)
                ])
        return entropyList
