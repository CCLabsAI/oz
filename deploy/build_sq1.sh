#!/bin/bash

#experiment name and version
experiment_name="sq1"
v_oz_test="0.0.31"
export TEST_OZ_NAME="gcr.io/ornate-axiom-187403/oz-test-${experiment_name}:$v_oz_test"
export SCRIPT_NAME='sq1'

CURR_PATH=`pwd`
SCRIPT_PATH="$CURR_PATH/deploy/build.sh"

#call build
bash "$SCRIPT_PATH"
