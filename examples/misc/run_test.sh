#!/bin/bash

#... { seq $N_CORE | parallel test.py } > out.txt

for n in `seq $N_CORE`; do
  test.py > out$n.txt &
  PIDS="$PIDS $$"
done

for p in $PIDS; do
  wait $p
done

python upload.py
