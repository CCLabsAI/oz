EXPERIMENT=goof6-test-10k

SRC_DIR=$HOME/src/oz
PYTHON=$HOME/anaconda3/bin/python
CHECKPOINT_DIR=$HOME/data/exp/$EXPERIMENT
LOG_PATH=$HOME/data/exp/${EXPERIMENT}.txt

OMP_NUM_THREADS=1 \
PYTHONPATH=$SRC_DIR \
$PYTHON $SRC_DIR/script/train.py \
	--game=goofspiel \
	--goofcards=6 \
	--checkpoint_dir=$CHECKPOINT_DIR \
	--checkpoint_interval=1 \
	--train_iter=2500 \
	--train_game_ply=6 \
	--train_batch_size=$[6*32] \
	--train_steps=128 \
	--nn_arch=deep \
	--hidden_sizes="128:64" \
	--search_batch_size=20 \
	--reservoir_size=$[1000*6*32] \
	--reservoir_beta_ratio=2.0 \
	--simulation_iter=10000 \
	--progress \
