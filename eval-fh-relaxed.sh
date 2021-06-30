#!/usr/bin/env bash

#input=input/*-100.pcap
#input=input/*-1000.pcap
#input="input/*-100.pcap input/*-1000.pcap"
#input=input/maxdiff-filtered/*-1000.pcap
# input=input/maxdiff-fromOrig/*-1000.pcap
#input=input/maxdiff-fromOrig/*-100.pcap
input="input/maxdiff-fromOrig/*-100*.pcap input/deduped-orig/*-100*.pcap"


#tftnext=$(expr 1 + $(ls -d reports/tft-* | sed "s/^.*tft-\([0-9]*\)-.*$/\1/" | sort | tail -1))
#tftnpad=$(printf "%03d" ${tftnext})
#currcomm=$(git log -1 --format="%h")
#report=reports/tft-${tftnpad}-clustering-${currcomm}
#mkdir ${report}

for fn in ${input} ; do
  python src/fh_relaxed.py ${fn}
  # Give tshark some time to recover
  sleep 3
done

#mv reports/*.csv ${report}/
#mv reports/*.pdf ${report}/

spd-say "Bin fertig!"
