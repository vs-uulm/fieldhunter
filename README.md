# FieldHunter

Re-implementation of the protocol reverse engineering approach FieldHunter as proposed in 

> Bermudez, Ignacio, Alok Tongaonkar, Marios Iliofotou, Marco Mellia, und Maurizio M. Munafò. 
> „Towards Automatic Protocol Field Inference“. Computer Communications 84 (15. Juni 2016). 
> https://doi.org/10.1016/j.comcom.2016.02.015.

Written by Stephan Kleber <stephan.kleber@uni-ulm.de>
who also proposed some improvements for the field heuristics in 
`inference/fieldtypesRelaxed.py`
used by
`src/fh_relaxed.py`
for evaluation to be run by
`eval-fh-relaxed.sh`.

The original FieldHunter heuristics are run via
`eval-fh.sh`.

Statistics about traces can be gained by
`eval-traces.sh`.

Not sure about a licence right now.

