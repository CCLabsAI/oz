EXPERIMENT=leduk-a-100k

BASE_DIR=/mnt
SRC_DIR=/mnt/oz/src
PYTHON=/mnt/miniconda3/bin/python
CHECKPOINT_DIR=/mnt/exp/$EXPERIMENT
LOG_PATH=/mnt/exp/${EXPERIMENT}.txt

OMP_NUM_THREADS=1 \
PYTHONPATH=$SRC_DIR \
nohup $PYTHON $SRC_DIR/script/train.py \
	--game leduk \
	--dist --workers=4 \
	--checkpoint_dir=$CHECKPOINT_DIR \
	--checkpoint_interval=1 \
	--train_iter=2500 \
	--train_game_ply=16 \
	--train_batch_size=$[4*16] \
	--train_steps=128 \
	--nn_arch=mlp \
	--hidden_size=64 \
	--search_batch_size=20 \
	--reservoir_size=$[2**16] \
	--reservoir_beta_ratio=2.0 \
	--simulation_iter=100000 \
	--print_ex --progress \
	>>$LOG_PATH 2>&1 <&- &
