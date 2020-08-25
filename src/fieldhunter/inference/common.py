"""
Common handling of inference intermediates or results.
"""

from typing import Iterable, List

from netzob.Model.Vocabulary.Messages.AbstractMessage import AbstractMessage

from fieldhunter.inference.fieldtypes import FieldType
from nemere.inference.formatRefinement import isOverlapping
from nemere.inference.segmentHandler import symbolsFromSegments
from nemere.inference.segments import TypedSegment, MessageSegment
from nemere.inference.analyzers import Value


def segmentedMessagesAndSymbols(typedFields: Iterable[FieldType], messages: List[AbstractMessage]):
    """
    Consolidate the inferred fields into segmented messages and additionally into symbols.

    :param typedFields: The inferred fields of different types in order of their precedence! E. g., field types with
        smaller index will remove concurring subsequent ones that overlap.
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