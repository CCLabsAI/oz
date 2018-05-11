#!/bin/bash

echo "---------------start-------------- $N_CORE"
echo "-command- python /app/examples/misc/test.py --game goofspiel --goofcards=$CARDS --p1 $P1 --p2 $P2 --iter1 $INTER_1 --iter2 $INTER_2 --eps 0.4 --delta 0.9 --gamma=0.01 --beta=0.99 > output/out$n.txt"

for n in `seq $N_CORE`; do
  python /app/examples/misc/test.py --game goofspiel --goofcards=$CARDS --p1 $P1 --p2 $P2 --iter1 $INTER_1 --iter2 $INTER_2 --eps 0.4 --delta 0.9 --gamma=0.01 --beta=0.99 > output/out$n.txt &
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