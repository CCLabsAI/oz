#!/bin/bash

echo "---------------start-------------- $N_CORE"

for n in `seq $N_CORE`; do
  python /app/examples/misc/test.py --game goofspiel --goofcards=13 --p1 oos_targeted --p2 oos_targeted --iter1 250000 --iter2 250000 --eps 0.4 --delta 0.9 --gamma=0.01 --beta=0.99 > output/out$n.txt &
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