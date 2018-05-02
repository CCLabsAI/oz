#!/bin/bash
DIR=`pwd`
echo $DIR

FULL_DIRECTORY="$DIR/dist"
echo "FULL_DIRECTORY: $FULL_DIRECTORY"

# Delete dist directory and create it again
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

echo 'wheel' + $wheel

docker build --build-arg wheel_filename=$wheel -f DockerfilePytorch -t ozpytorch .

docker rm ozpytorch1

docker run --name ozpytorch1 -ti ozpytorch