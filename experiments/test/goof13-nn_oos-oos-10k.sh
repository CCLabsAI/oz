EXPERIMENT=goof13-nn_oos-oos-10k
EXP_DIR=/mnt/exp/test/$EXPERIMENT
OZ_DIR=/mnt/src/oz
PYTHON=/mnt/miniconda3/bin/python
CHECKPOINT=/mnt/exp/goof13-a-10k/checkpoint-000500.pth
NJOBS=250

mkdir -p $EXP_DIR

seq $NJOBS | \
OMP_NUM_THREADS=1 \
PYTHONPATH="/mnt/src/oz" \
parallel --eta $PYTHON examples/misc/test.py \
	--matches=20 \
	--game=goofspiel2 \
	--goofcards=13 \
	--p1=nn_oos \
	--iter1=13000 \
	--checkpoint_path1=$CHECKPOINT \
	--search_batch_size=10 \
	--p2=oos_targeted \
	--iter2=10000 \
	">" $EXP_DIR/output-{#}.log
