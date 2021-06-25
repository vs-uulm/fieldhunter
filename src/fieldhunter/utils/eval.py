
from nemere import MessageComparator

from fieldhunter.inference.fieldtypes import *



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


