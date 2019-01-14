#!/bin/sh

#OPEN=open
OPEN=xdg-open

# GRAPHVIZ=dot
GRAPHVIZ=neato

python tools/game2dot.py > /tmp/game.dot && \
	$GRAPHVIZ -Tpdf /tmp/game.dot > /tmp/game.pdf && \
	$OPEN /tmp/game.pdf

$GRAPHVIZ -Tsvg /tmp/game.dot > /tmp/game.svg
