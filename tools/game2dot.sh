#!/bin/sh

python tools/game2dot.py > /tmp/game.dot && \
	dot -Tpdf /tmp/game.dot > /tmp/game.pdf && \
	open /tmp/game.pdf
