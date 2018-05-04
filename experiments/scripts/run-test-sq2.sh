#!/bin/bash

echo "---------------start-------------- "

for n in `seq $N_CORE`; do
  python /app/examples/misc/test.py --game goofspiel --goofcards=6 --p1 oos_targeted --p2 random --iter1 100 --iter2 0 --eps 0.4 --delta 0.9 --gamma=0.01 > output/out$n.txt &
done

wait

echo "---------------end----------------"

OUTPUT="$(ls output -1)"
echo "${OUTPUT}"


echo "$(pwd)"
STORAGE_PATH="/app/experiments/scripts/storage/storage.py"
CONFIG_PATH="/app/experiments/scripts/storage/cclabs.gcp.json"
OUTPUT_PATH="/app/output/"
SEARCH_PATH="$OUTPUT_PATH*"

for file in $SEARCH_PATH; do
    echo "FOUND $file"
    python  $STORAGE_PATH -c $CONFIG_PATH -f "$file" -n "sq1"
done