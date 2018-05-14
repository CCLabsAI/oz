#!/bin/bash

echo "---------------start-------------- $N_CORE"
echo "$(ls /mnt/data/exp/goof13-a-100k/ )"

CMD_CHECKPOINT_1=""
CMD_CHECKPOINT_2=""
if [ "$CHECKPOINT_1" != "" ]; then
  CMD_CHECKPOINT_1="--checkpoint_path1 $CHECKPOINT_1 "
fi

if [ "$CHECKPOINT_2" != "" ]; then
  CMD_CHECKPOINT_2="--checkpoint_path2 $CHECKPOINT_2 "
fi

echo "CMD_CHECKPOINT_1: $CMD_CHECKPOINT_1     CMD_CHECKPOINT_2: $CMD_CHECKPOINT_2"

echo "-command- python /app/examples/misc/test.py --game goofspiel --goofcards=$CARDS --p1 $P1 $CMD_CHECKPOINT_1 --iter1 $INTER_1 --p2 $P2 $CMD_CHECKPOINT_2 --iter2 $INTER_2 --eps 0.4 --delta 0.9 --gamma=0.01 --beta=0.99 > output/out$n.txt"


for n in `seq $N_CORE`; do
  python /app/examples/misc/test.py --game goofspiel --goofcards=$CARDS --p1 $P1 $CMD_CHECKPOINT_1 --iter1 $INTER_1 --p2 $P2 $CMD_CHECKPOINT_2 --iter2 $INTER_2 --eps 0.4 --delta 0.9 --gamma=0.01 --beta=0.99 > output/out$n.txt &
done

wait

echo "---------------end command----------------"

OUTPUT="$(ls output -1)"
echo "${OUTPUT}"


echo "$(pwd)"
STORAGE_PATH="/app/experiments/scripts/storage/storage.py"
CONFIG_PATH="/app/experiments/scripts/storage/cclabs.gcp.json"
OUTPUT_PATH="/app/output/"
SEARCH_PATH="$OUTPUT_PATH*"

for file in $SEARCH_PATH; do
    echo "FOUND $file"
    python  $STORAGE_PATH -c $CONFIG_PATH -f "$file" -n "sq1" -b "$FOLDER"
done

echo "---------------end save files----------------"