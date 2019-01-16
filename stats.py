import argparse
from os import walk
from os.path import splitext
from os.path import join
import numpy as np
import sys



# Script to calculate and print Stats for NIPS paper
# Files .txt collected in a folder (to be specified as second argument path)
# Leduc Goofspield and Liar's dice games tested


# Example usage for poker
# python stats.py --game leduk --path out_leduk

# Usage


def main():

    game_type = 0;
    parser = argparse.ArgumentParser(description="run statistics on results")
    parser.add_argument("--game", help="game to play", required=True)
    parser.add_argument("--path", help="txt file with the results", required=True)
    args = parser.parse_args()

    if args.game == 'leduk' or args.game == 'leduk_poker':
        print('Statistics for Leduk')
        game_type = 1
    elif args.game == 'goofspiel' or args.game == 'goofspiel2':
        print('Statistics for Goofspiel')
        game_type = 2
    elif args.game == 'liars_dice' :
        print('Statistics for Liars Dice')
        game_type = 2
    else:
        print('error: unknown game: {}'.format(args.game), file=sys.stderr)
        exit(1)


    path = args.path
    first_line = True

    all_results = []
    count_files = 0
    tot_matches = 0

    print(path)
    for root, dirs, files in walk(path):
        for f in sorted(files):
            if splitext(f)[1].lower() == ".txt":
                filename = join(root, f)
                print()
                print('Reading ', filename)
                print()
                result_value = []


                with open(filename,"r") as file:
                    count_files += 1
                    for line in file:

                        if line.startswith("R"):
                            string, value = line.rstrip().split(':')
                            if(game_type == 1 ):
                                result_value.append(int(value))
                                all_results.append(int(value))
                            else :
                                result_value.append(int(value))
                                if(int(value) == 1):
                                    all_results.append(1.0)
                                elif(int(value) == 0):
                                    all_results.append(0.5)
                                else :
                                    all_results.append(0.0)
                        if line.startswith("label "):
                            name,hash_value = line.rstrip().split(':')
                        elif line.startswith("Iters"):
                            name, iter1, iter2 = line.rstrip().split(' ')
                        elif line.startswith("Checkpoints"):
                            name, ckp1, ckp2 = line.rstrip().split(' ')
                        elif line.startswith("Execution "):
                            name1, name2, exec_time = line.rstrip().split(' ')
                            exec_time = int(exec_time)
                        elif line.startswith("N"):
                            n_matches = int(line[1:])
                            tot_matches += n_matches
                        elif line.startswith("Players"):
                            p0, p1, p2 = line.rstrip().split(' ')
                        elif line.startswith("Cards"):
                            name,cards_number = line.rstrip().split(':')
                            cards_number = int(cards_number)
                        elif line.startswith("epsilon"):
                            name, epsilon = line.rstrip().split(':')
                        elif line.startswith("beta"):
                            name, beta = line.rstrip().split(':')
                        elif line.startswith("delta"):
                             name, delta = line.rstrip().split(':')
                        elif line.startswith("gamma"):
                             name, gamma = line.rstrip().split(':')

                    assert len(result_value) == n_matches, "Number of results must be equal to the number of match"

                    # Calculate percentage of win

                    count_win = 0
                    count_lose = 0
                    count_draw = 0

                    if(game_type == 1):
                        for i in range(len(result_value)):
                            if(result_value[i] > 0):
                               count_win += 1

                            elif(result_value[i] < 0):
                                count_lose += 1

                            elif(result_value[i] == 0):
                                count_draw += 1
                    else :
                        for i in range(len(result_value)):
                            if(result_value[i] == 1):
                                count_win += 1
                            elif(result_value[i] == -1):
                                count_lose += 1
                            else :
                                count_draw += 1

                    percentage_win = count_win / len(result_value);
                    percentage_lose = count_lose / len(result_value);
                    percentage_draw = count_draw / len(result_value);

                    #print('git hash value ', hash_value)

                    print("Winning % : ", percentage_win)
                    print("Losing % : ", percentage_lose)
                    print("Drawing % : ", percentage_draw)
                    print("Total number of matches ", n_matches)

                    print()
                    print("------------------------------------------------------------------------")
                    print()




        print("               Final result ")
        print()
        print()
        print("Git hash                  : ", hash_value)
        print("Game                      : ", args.game)
        print("Number of matches played  : ", tot_matches)
        print("Number of cards           : ", cards_number)
        print("Players                   :  P1 :", p1, "    P2 :", p2)
        print("Iterations for P1         :", iter1)
        print("Iterations for P2         :", iter2)
        print("Checkpoint for P1         :", ckp1)
        print("Checkpoint for P2         :", ckp2)
        print()
        print("Execution time for each task (80 match) : ", (exec_time / (1000 * 60)), "minutes")
        print("Number of tasks executed : ", count_files)
        print()
        print()
        print("        Parameters :")
        print()
        print("beta    :", beta)
        print("delta   :", delta)
        print("epsilon :", epsilon)
        print("gamma   :", gamma)

        print()

        tot_mean = np.mean(all_results)
        if(game_type == 1):
            print("Average result :", (tot_mean) )
        else :
            print("Winning Rate :", (tot_mean) * 100, " %" )

        tot_std = np.std(all_results)


        # 95% Confidence Interval

        z = 1.960

        confidence_interval_lower = tot_mean - z * (tot_std / np.sqrt(len(all_results)))
        confidence_interval_upper = tot_mean + z * (tot_std / np.sqrt(len(all_results)))

        if(game_type == 1):
            print('95% Confidence Interval :',  (confidence_interval_upper - confidence_interval_lower))
        else :
            print('95% Confidence Interval :',  (confidence_interval_upper - confidence_interval_lower) * 100)
        print()
        print()


if __name__ == "__main__":
    # execute only if run as a script
    main()
