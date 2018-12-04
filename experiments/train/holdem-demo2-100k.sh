EXPERIMENT=holdem-demo2-100k

SRC_DIR=/mnt/src/oz
PYTHON=/mnt/miniconda3/bin/python
CHECKPOINT_DIR=/mnt/exp/$EXPERIMENT
LOG_PATH=/mnt/exp/${EXPERIMENT}.txt

OMP_NUM_THREADS=1 \
PYTHONPATH=$SRC_DIR \
nohup nice $PYTHON $SRC_DIR/script/train.py \
	--game=holdem \
	--checkpoint_dir=$CHECKPOINT_DIR \
	--dist --workers=64 \
	--checkpoint_interval=1 \
	--train_iter=5000 \
	--train_game_ply=16 \
	--train_batch_size=$[16*32] \
	--train_steps=8 \
	--nn_arch=holdem_demo \
	--search_batch_size=20 \
	--reservoir_size=$[2*22] \
	--reservoir_beta_ratio=2.0 \
	--simulation_iter=100000 \
	--learning_rate=1e-4 \
	--progress \
	>>$LOG_PATH 2>&1 <&- &
