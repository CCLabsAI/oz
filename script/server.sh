#!/usr/bin/env bash

FILE="script/server.py"

trap "exit" INT TERM ERR
trap "kill 0" EXIT

RUN="python $FILE"
WAIT="inotifywait"
WATCH="$FILE"

if ! type -P $WAIT >/dev/null 2>&1; then
	echo "$0: WARNING: reloading disabled, $WAIT not found" >&2
	exec $RUN
fi

$RUN &
while $WAIT -e modify "$WATCH"; do
	kill $!
	$RUN &
done
