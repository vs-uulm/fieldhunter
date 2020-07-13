"""
This script provides statistics about the given PCAP trace that have impact on the FieldHunter inference.
"""


# Relevant for MSG-Type
# TODO amount/percentage of messages in the trace that are of singular flows,
#   i. e. without a matching request or reply
#   discern types: broadcasts, c2s/s2c without matching flow
#
# Entropy filter threshold rationale -> e. g. some histogram, CDF, ...
#


# Relevant for MSG-Len
# TODO length of messages, something like:
#         keyfunc = lambda m: len(m.data)
#         msgbylen = {k: v for k, v in groupby(sorted(direction, key=keyfunc), keyfunc)}