EXPERIMENT=holdem-test2-10k

SRC_DIR=$HOME/src/oz
PYTHON=$HOME/anaconda3/bin/python
CHECKPOINT_DIR=$HOME/data/exp/$EXPERIMENT
LOG_PATH=$HOME/data/exp/${EXPERIMENT}.txt

OMP_NUM_THREADS=1 \
PYTHONPATH=$SRC_DIR \
nohup nice $PYTHON $SRC_DIR/script/train.py \
	--game=holdem \
	--checkpoint_dir=$CHECKPOINT_DIR \
	--pretrained_model=$HOME/src/poker-predict/models/poker_action_model_demo1.pth \
	--checkpoint_interval=1 \
	--train_iter=1000 \
	--train_game_ply=16 \
	--train_batch_size=$[16*32] \
	--train_steps=8 \
	--nn_arch=holdem_demo \
	--search_batch_size=20 \
	--reservoir_size=$[2*22] \
	--reservoir_beta_ratio=2.0 \
	--simulation_iter=10000 \
	--learning_rate=1e-4 \
	--progress \
	>>$LOG_PATH 2>&1 <&- &
