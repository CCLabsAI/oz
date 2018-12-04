import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.utils.data
import torch.distributed as dist
from torch.multiprocessing import Process

import oz
import oz.reservoir
import oz.dist
from oz.train import Trainer
import oz.nn

import os
import argparse
import sys
import signal
import pprint

def main():
    parser = argparse.ArgumentParser(description="train NN with OmegaZero")
    parser.add_argument("--game", help="game to play", required=True)

    group = parser.add_mutually_exclusive_group()
    group.add_argument("--dist", dest="dist", action="store_true")
    group.add_argument("--no-dist", dest="dist", action="store_false")
    parser.set_defaults(dist=False)

    parser.add_argument("--workers", type=int,
                        help="number of workers",
                        default=8)

    group = parser.add_mutually_exclusive_group()
    group.add_argument("--print_ex", dest="print_ex", action="store_true")
    group.add_argument("--no-print_ex", dest="print_ex", action="store_false")
    parser.set_defaults(print_ex=False)

    group = parser.add_mutually_exclusive_group()
    group.add_argument("--progress", dest="progress", action="store_true")
    group.add_argument("--no-progress", dest="progress", action="store_false")
    parser.set_defaults(progress=False)

    parser.add_argument("--nn_arch",
                        help="nn architecture",
                        choices=["mlp", "deep", "holdem_demo"],
                        default="mlp")
    # parser.add_argument("--opt",
    #                     help="optimizer to use",
    #                     choices=["sgd", "adam"],
    #                     default="adam")
    parser.add_argument("--hidden_size", type=int,
                        help="mlp hidden layer size",
                        default=64)
    parser.add_argument("--hidden_sizes",
                        help="deep hidden layer sizes, e.g. (256:256:128)",
                        default="64:64")

    parser.add_argument("--checkpoint_dir", required=True,
                        help="checkpoint directory")
    parser.add_argument("--checkpoint_interval", type=int,
                        help="number of iterations between checkpoints",
                        default=25)

    parser.add_argument("--goofcards", type=int,
                        help="number of cards for II Goofspiel",
                        default=6)
    parser.add_argument("--learning_rate", type=float,
                        help="learning rate",
                        default=1e-3)
    parser.add_argument("--eps", type=float,
                        help="exploration factor",
                        default=0.4)
    parser.add_argument("--delta", type=float,
                        help="targeting factor",
                        default=0.9)
    parser.add_argument("--gamma", type=float,
                        help="opponent error factor",
                        default=0.01)
    parser.add_argument("--beta", type=float,
                        help="reweighting moving average decay rate",
                        default=0.99)
    parser.add_argument("--eta", type=float,
                        help="currently unused",
                        default=1.0)
    parser.add_argument("--search_batch_size", type=int,
                        help="search batch size",
                        default=20)
    parser.add_argument("--play_eps", type=float,
                        help="playout eps",
                        default=0.2)
    parser.add_argument("--simulation_iter", type=int,
                        help="simulation iterations",
                        default=5000)
    parser.add_argument("--reservoir_size", type=int,
                        help="reservoir size",
                        default=2**17)
    parser.add_argument("--reservoir_beta_ratio", type=float,
                        help="simulation iterations",
                        default=2.0)
    parser.add_argument("--train_game_ply", type=int,
                        help="number of plies between nn training",
                        default=128)
    parser.add_argument("--train_batch_size", type=int,
                        help="nn training batch size",
                        default=128)
    parser.add_argument("--train_steps", type=int,
                        help="nn training steps per iteration",
                        default=256)
    parser.add_argument("--train_iter", type=int,
                        help="nn training iterations",
                        default=1000)
    parser.add_argument("--pretrained_model",
                        help="holdem pretrained checkpoint to load")

    args = parser.parse_args()

    if os.path.exists(args.checkpoint_dir):
        files = [f for f in os.listdir(args.checkpoint_dir)
                  if not f.startswith('.')]

        if files:
            files.sort()
            latest_checkpoint_name = files[-1]
            checkpoint_path = os.path.join(args.checkpoint_dir,
                                           latest_checkpoint_name)
            print('loading checkpoint: {}'.format(checkpoint_path))
            ob = torch.load(checkpoint_path)
            checkpoint_args = argparse.Namespace(**ob['args'])
            # TODO print warning if loaded args differ
            if args != checkpoint_args:
                print('WARNING: checkpoint args differ from command line args')
            run(checkpoint_args, checkpoint_data=ob)
        else:
            run(args)

    else:
        os.makedirs(args.checkpoint_dir)
        run(args)

def save_checkpoint(iteration_n, trainer, args):
    ob = {
        'iteration_n': iteration_n,
        'args': vars(args),
        'model_state': trainer.model.state_dict(),
        'optimizer_state': trainer.optimizer.state_dict()
    }

    dir_path = args.checkpoint_dir
    checkpoint_name = 'checkpoint-{:06d}.pth'.format(iteration_n)
    full_path = os.path.join(dir_path, checkpoint_name)
    torch.save(ob, full_path)
    print('saved checkpoint: {}'.format(full_path))

def run(args, checkpoint_data=None):
    print(vars(args))

    game = args.game
    if game == 'leduc' or game == 'leduc_poker' or \
       game == 'leduk' or game == 'leduk_poker':
        history = oz.make_leduk_history()
        encoder = oz.make_leduk_encoder()
        target  = oz.make_leduk_target()
    elif game == 'goofspiel' or game == 'goofspiel2':
        n_cards = args.goofcards
        history = oz.make_goofspiel2_history(n_cards)
        encoder = oz.make_goofspiel2_encoder(n_cards)
        target  = oz.make_goofspiel2_target()
    elif game == 'holdem' or game == 'holdem_poker':
        history = oz.make_holdem_history()
        encoder = oz.make_holdem_encoder()
        target  = oz.make_holdem_target()
    else:
        raise "unknown game: {}".format(args.game)

    rng = oz.Random()

    model = oz.nn.model_with_args(args,
                input_size=encoder.encoding_size(),
                output_size=encoder.max_actions())

    if args.pretrained_model and not checkpoint_data:
        print('loading pretrained model: {}...'.format(args.pretrained_model))
        state_dict = torch.load(args.pretrained_model, map_location='cpu')
        model.load_state_dict(state_dict)

    def make_batch_search():
        return oz.BatchSearch(batch_size=args.search_batch_size,
                              history=history,
                              encoder=encoder,
                              target=target,
                              eps=args.eps, delta=args.delta, gamma=args.gamma,
                              beta=args.beta, eta=args.eta)

    optimizer = optim.Adam(model.parameters(), lr=args.learning_rate)
    # optimizer = optim.SGD(model.parameters(), lr=learning_rate)
    criterion = nn.KLDivLoss(size_average=False)

    trainer = Trainer(history=history,
                      model=model,
                      make_batch_search=make_batch_search,
                      encoder=encoder,
                      optimizer=optimizer,
                      criterion=criterion,
                      simulation_iter=args.simulation_iter,
                      play_eps=args.play_eps,
                      rng=rng)

    # TODO make this more efficient
    def pr_nn(infoset, action):
        enc = trainer.encoder
        d = torch.zeros(enc.encoding_size())
        enc.encode(infoset, d)
        sigma_logits = trainer.model.forward(d.unsqueeze(0))
        sigma_pr = F.softmax(sigma_logits, dim=1)
        m = enc.decode(infoset, sigma_pr.data[0])
        total = sum(m.values())
        return m[action] / total

    sigma_nn = oz.make_py_sigma(pr_nn)

    def iter_callback(iteration_n, interrupted, trainer, args, losses):
        if not interrupted:
            if args.print_ex:
                ex = oz.exploitability(trainer.root_history, sigma_nn)
                print('ex: {:.5f}'.format(ex), flush=True)

            mean_loss = losses.mean()
            print('mean loss: {:.5f}'.format(mean_loss.item()), flush=True)

            if iteration_n % args.checkpoint_interval == 0:
                save_checkpoint(iteration_n=iteration_n,
                                trainer=trainer,
                                args=args)

        else:
            save_checkpoint(iteration_n=iteration_n,
                            trainer=trainer,
                            args=args)

    # load params from checkpoint
    if checkpoint_data:
        start_iteration = checkpoint_data['iteration_n']
        model.load_state_dict(checkpoint_data['model_state'])
        optimizer.load_state_dict(checkpoint_data['optimizer_state'])
    else:
        start_iteration = 0

    if args.dist:
        oz.dist.run_trainer_distributed(trainer, args,
            size=args.workers,
            start_iteration=start_iteration,
            iter_callback=iter_callback)
    else:
        def sigint_handler(signal, frame):
            oz.train.interrupted = True
        signal.signal(signal.SIGINT, sigint_handler)

        oz.train.run_trainer_reservoir(trainer, args,
            start_iteration=start_iteration,
            iter_callback=iter_callback)

if __name__ == "__main__":
    main()
