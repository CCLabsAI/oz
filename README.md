# ExIt-OOS

## Description 
ExIt-OOS is a learning algorithm that enables the computer to learn how to play a variety of imperfect information games through self-play. Please find full description of the algorithm at https://arxiv.org/abs/1808.10120


## To install:

```shell
$ python setup.py
```

To build in debug mode and install in development mode:

```shell
$ python setup.py build --debug develop
```

## Dependencies:

You will need to install pytorch 1.0


## Usage 

To train the model run:

```shell
$ python python script/train.py --game [name of the game] --checkpoint_dir [name of the dir]
```


To see the agent playing against itself:

```shell
$ python examples/misc/test.py --game [name of the game] --p1 [type of player 1 for example random or oos ] --p2 [type of player 2]
```


## License 

See LICENSE file
