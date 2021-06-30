"""
Common handling of inference intermediates or results.
"""

from typing import Iterable, List, Tuple, Dict

from netzob.Model.Vocabulary.Messages.AbstractMessage import AbstractMessage
from netzob.Model.Vocabulary.Symbol import Symbol

from fieldhunter.inference.fieldtypes import FieldType
from nemere.inference.formatRefinement import isOverlapping
from nemere.inference.segmentHandler import symbolsFromSegments
from nemere.inference.segments import TypedSegment, MessageSegment
from nemere.inference.analyzers import Value


def segmentedMessagesAndSymbols(typedFields: Iterable[FieldType], messages: List[AbstractMessage]) \
        -> Tuple[Dict[AbstractMessage, List[MessageSegment]], List[Symbol]]:
    # noinspection PyProtectedMember
    """
    Consolidate the inferred fields into segmented messages and additionally into symbols.

    >>> from itertools import chain
    >>> from tabulate import tabulate
    >>> from netzob.Model.Vocabulary.Messages.L4NetworkMessage import L4NetworkMessage
    >>> from nemere.visualization.simplePrint import SegmentPrinter
    >>> from fieldhunter.inference.common import segmentedMessagesAndSymbols
    >>> from fieldhunter.inference.fieldtypes import FieldType
    >>> from fieldhunter.utils.base import iterateSelected
    >>> # prevent Netzob from producing debug output.
    >>> import logging
    >>> logging.getLogger().setLevel(30)
    >>>
    >>> messageList = [
    ...    L4NetworkMessage(b"QQQ456789"), L4NetworkMessage(b"RRR567890"), L4NetworkMessage(b"QQQ7890AB"),
    ...    L4NetworkMessage(b"RRR567890"), L4NetworkMessage(b"QQQ123456789"), L4NetworkMessage(b"RRR890A"),
    ...    L4NetworkMessage(b"QQQ6789"), L4NetworkMessage(b"RRR890ABCDEFGH")
    ... ]
    >>>
    >>> # normally this would only be performed by a subclass of FieldType internally; here for the sake of testing
    >>> segmentsA = FieldType._posLen2segments(messageList, [(0,3),(5,2)])
    >>> del segmentsA[5][1]; del segmentsA[3]; del segmentsA[1][1]; del segmentsA[0][0]
    >>> segmentsB = FieldType._posLen2segments(messageList, [(2,2),(5,4)])
    >>> ftA = FieldType()
    >>> ftA._segments = segmentsA
    >>> ftB = FieldType()
    >>> ftB._segments = segmentsB
    >>>
    >>> sm, sym = segmentedMessagesAndSymbols([ftA, ftB], messageList)
    >>> sp = SegmentPrinter(sm.values())  # doctest: +SKIP
    >>> sp.toConsole()  # doctest: +SKIP
    >>> print(tabulate(sm.values()))
    ---------------------------------------------------------------  -------------------------------------------------------------------
    MessageSegment 2 bytes at (2, 4): 5134 | values: (81, 52)        MessageSegment 2 bytes at (5, 7): 3637 | values: (54, 55)
    MessageSegment 3 bytes at (0, 3): 525252 | values: (82, 82, 82)  MessageSegment 4 bytes at (5, 9): 37383930 | values: (55, 56, 57...
    MessageSegment 3 bytes at (0, 3): 515151 | values: (81, 81, 81)  MessageSegment 2 bytes at (5, 7): 3930 | values: (57, 48)
    MessageSegment 2 bytes at (2, 4): 5235 | values: (82, 53)        MessageSegment 4 bytes at (5, 9): 37383930 | values: (55, 56, 57...
    MessageSegment 3 bytes at (0, 3): 515151 | values: (81, 81, 81)  MessageSegment 2 bytes at (5, 7): 3334 | values: (51, 52)
    MessageSegment 3 bytes at (0, 3): 525252 | values: (82, 82, 82)
    MessageSegment 3 bytes at (0, 3): 515151 | values: (81, 81, 81)  MessageSegment 2 bytes at (5, 7): 3839 | values: (56, 57)
    MessageSegment 3 bytes at (0, 3): 525252 | values: (82, 82, 82)  MessageSegment 2 bytes at (5, 7): 3041 | values: (48, 65)
    ---------------------------------------------------------------  -------------------------------------------------------------------
    >>> for s in sym:
    ...     print(s.getCells())  # doctest: +NORMALIZE_WHITESPACE
    Field | Field | Field | Field | Field
    ----- | ----- | ----- | ----- | -----
    'QQ'  | 'Q4'  | '5'   | '67'  | '89'
    ----- | ----- | ----- | ----- | -----
    Field | Field | Field
    ----- | ----- | ------
    'RRR' | '56'  | '7890'
    ----- | ----- | ------
    Field | Field | Field | Field
    ----- | ----- | ----- | -----
    'QQQ' | '78'  | '90'  | 'AB'
    ----- | ----- | ----- | -----
    Field | Field | Field | Field
    ----- | ----- | ----- | ------
    'RR'  | 'R5'  | '6'   | '7890'
    ----- | ----- | ----- | ------
    Field | Field | Field | Field
    ----- | ----- | ----- | -------
    'QQQ' | '12'  | '34'  | '56789'
    ----- | ----- | ----- | -------
    Field | Field
    ----- | ------
    'RRR' | '890A'
    ----- | ------
    Field | Field | Field
    ----- | ----- | -----
    'QQQ' | '67'  | '89'
    ----- | ----- | -----
    Field | Field | Field | Field
    ----- | ----- | ----- | ---------
    'RRR' | '89'  | '0A'  | 'BCDEFGH'
    ----- | ----- | ----- | ---------

    :param typedFields: The inferred fields of different types in order of their precedence!
        E. g., field types with smaller index will remove concurring subsequent ones that overlap.
    :param messages: The messages to expect inference for.
    :return: tuple of
        * dict of the messages and their segment list.
        * Netzob symbols representing the inference.
    """
    # combine inferred fields per message to facilitate validation
    typedSequences = [
        {segs[0].message: segs for segs in fields.segments if len(segs) > 0}
        for fields in typedFields
    ]

    segmentedMessages = dict()
    for msg in messages:
        segmsg = list()
        # in order of fieldtypes.precedence!
        for typedMessages in typedSequences:
            if msg in typedMessages:  # type: List[TypedSegment]
                # segments of a field type for one message
                for cand in typedMessages[msg]:
                    # check overlapping segment
                    overlapps = False
                    for seg in segmsg:
                        if isOverlapping(cand, seg):
                            overlapps = True
                            break
                    # if a segment is already
                    if overlapps:
                        continue
                    segmsg.append(cand)
        # symbolsFromSegments fixes gaps, but cannot know anything about the message in an empty list, so we add a dummy
        # segment for these cases here
        segmentedMessages[msg] = sorted(segmsg, key=lambda s: s.offset) if len(segmsg) > 0 else \
            [ MessageSegment(Value(msg), 0, len(msg.data)) ]

    symbols = symbolsFromSegments(segmentedMessages.values())

    return segmentedMessages, symbols