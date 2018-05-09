#!/bin/bash
DIR=`pwd`

FULL_DIRECTORY="$DIR/dist"

#Delete dist directory and create it again
if [ -d "$FULL_DIRECTORY" ]; then
  rm -r $FULL_DIRECTORY
fi

mkdir $FULL_DIRECTORY

#build
docker build -f DockerfileWheel -t ozwheel .
docker rm ozwheel1
docker run -v $FULL_DIRECTORY:/app/dist --name ozwheel1 -ti ozwheel


SEARCH_DIR="$FULL_DIRECTORY/*"

for file in $SEARCH_DIR; do
    wheel=$(basename "$file")
done

git_hash=$(git rev-parse HEAD 2>&1)

docker build --build-arg wheel_filename=$wheel -f DockerfilePytorch -t $TEST_OZ_NAME .

echo "docker build --build-arg wheel_filename=$wheel -f DockerfilePytorch -t $TEST_OZ_NAME ."

# docker rm ozpytorch1
# docker run -e GIT_HASH=$git_hash -e SCRIPT_NAME=$SCRIPT_NAME --name ozpytorch1 -ti $TEST_OZ_NAME