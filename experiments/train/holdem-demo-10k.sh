EXPERIMENT=holdem-demo-10k

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
	--train_iter=2500 \
	--train_game_ply=16 \
	--train_batch_size=$[16*32] \
	--train_steps=512 \
	--nn_arch=deep \
	--hidden_sizes="256:256:128" \
	--search_batch_size=20 \
	--reservoir_size=$[2**22] \
	--reservoir_beta_ratio=2.0 \
	--simulation_iter=10000 \
	--progress \
	>>$LOG_PATH 2>&1 <&- &
