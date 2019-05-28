import argparse
from os import listdir
import os
import matplotlib.pyplot as plt
import matplotlib
import torch
import torch.nn as nn
import numpy as np

parser = argparse.ArgumentParser(description="Jakub's hacked up project visualizer")
parser.add_argument('--dir', default="")
parser.add_argument('--layer_name', default="")
parser.add_argument('--scenario', default="")

# font = {'family' : 'normal',
#         'weight' : 'bold',
#         'size'   : 22}

# matplotlib.rc('font', **font)

def plot_singular_values_over_epochs(data_sets, layer_name, scenario):
    fig = plt.figure()
    fig.suptitle('Singular Values and Conditioning ('+scenario+')', fontsize=18)


    for key in data_sets.keys():
        data = data_sets[key]
        cond_list = []
        mins_list = []
        maxs_list = []
        accuracy_list = []
        for d in data:
            mins = min(d['stats'][layer_name]['singular_values'].numpy())
            maxs = max(d['stats'][layer_name]['singular_values'].numpy())
            cond = float(maxs)/float(mins)
            cond_list.append(cond)
            mins_list.append(mins)
            maxs_list.append(maxs)
            accuracy_list.append(d['curr_acc1'])

        plt.subplot(3, 1, 1)
        plt.grid(True)
        plt.xlabel('Epoch #', fontsize=18)
        plt.ylabel('Minimum singular value', fontsize=18)
        plt.yscale('log')
        plt.plot(mins_list, label=key)

        plt.subplot(3, 1, 2)
        plt.grid(True)

        plt.xlabel('Epoch #', fontsize=18)
        plt.ylabel('Maximum singular value', fontsize=18)
        plt.plot(maxs_list, label=key)

        plt.subplot(3, 1, 3)
        plt.grid(True)
        plt.xlabel('Epoch #', fontsize=18)
        plt.ylabel('Conditioning', fontsize=18)
        plt.yscale('log')
        # plt.ylim(10, 10e9)
        plt.plot(cond_list, label=key)

    plt.show(block=False)
    plt.legend(loc='upper left', frameon=False)

def plot_accuracy(data_sets, layer_name, scenario):
    fig = plt.figure()
    fig.suptitle('Conditioning and Accuracy (' +scenario+')', fontsize=18)
    # sucks this is copy pasted ):
    for key in data_sets.keys():
        data = data_sets[key]
        cond_list = []
        mins_list = []
        maxs_list = []
        accuracy_list = []
        for d in data:
            mins = min(d['stats'][layer_name]['singular_values'].numpy())
            maxs = max(d['stats'][layer_name]['singular_values'].numpy())
            cond = float(maxs)/float(mins)
            cond_list.append(cond)
            mins_list.append(mins)
            maxs_list.append(maxs)
            accuracy_list.append(d['curr_acc1'])

        diff_acc = [j-i for i, j in zip(accuracy_list[:-1], accuracy_list[1:])]


        print(key)
        print(max(diff_acc))
        print(min(diff_acc))
        print("The following are the max and min accuracies")
        print(max(accuracy_list))
        print(min(accuracy_list))

        plt.subplot(3, 1, 1)
        plt.grid(True)
        plt.xlabel('Epoch #', fontsize=18)
        plt.ylabel('Accuracy', fontsize=18)
        plt.ylim(0, 15)
        plt.plot(accuracy_list, label=key)

        plt.subplot(3, 1, 2)
        plt.grid(True)
        plt.xlabel('Epoch #', fontsize=18)
        plt.ylabel(u'\N{GREEK CAPITAL LETTER DELTA} Accuracy', fontsize=18)
        # plt.ylim(-1.0, 1.0)
        plt.plot(diff_acc, label=key)

        plt.subplot(3, 1, 3)
        plt.grid(True)
        plt.xlabel('Epoch #', fontsize=18)
        plt.ylabel('Conditioning', fontsize=18)
        plt.yscale('symlog')
        # plt.ylim(10, 10e5)
        plt.plot(cond_list, label=key)

    plt.show(block=False)
    plt.legend(loc='upper left')

def plot_singular_values(data_sets, layer_name):
    for key in data_sets.keys():
        fig = plt.figure()
        data = data_sets[key]
        cond_list = []
        mins_list = []
        maxs_list = []
        accuracy_list = []
        i = 0
        for d in data:
            plt.plot(np.cumsum(d['stats'][layer_name]['singular_values'].numpy()[::-1]))
            i = i + 1
            mins = min(d['stats'][layer_name]['singular_values'].numpy())
            maxs = max(d['stats'][layer_name]['singular_values'].numpy())
            cond = float(maxs)/float(mins)
            cond_list.append(cond)
            mins_list.append(mins)
            maxs_list.append(maxs)
            accuracy_list.append(d['curr_acc1'])
            plt.yscale('symlog')
        plt.show(block=False)



def plot_accuracy_v2(data_sets, layer_name, scenario):
    fig = plt.figure()
    fig.suptitle('Conditioning vs Accuracy (' +scenario +')', fontsize=18)

    # sucks this is copy pasted ):
    for key in data_sets.keys():
        data = data_sets[key]
        cond_list = []
        mins_list = []
        maxs_list = []
        accuracy_list = []
        for d in data:
            mins = min(d['stats'][layer_name]['singular_values'].numpy())
            maxs = max(d['stats'][layer_name]['singular_values'].numpy())
            cond = float(maxs)/float(mins)
            cond_list.append(cond)
            mins_list.append(mins)
            maxs_list.append(maxs)
            accuracy_list.append(d['curr_acc1'])

        # for i in range(0, len(cond_list)-1):
        plt.plot(cond_list, accuracy_list, label=key)
        plt.ylabel("Accuracy %", fontsize=18)
        plt.xlabel("Conditioning", fontsize=18)
        plt.grid(True)
        plt.xscale('symlog')
        plt.legend(loc='lower right')


    fig = plt.figure()
    fig.suptitle(u'\N{GREEK CAPITAL LETTER DELTA} Conditioning vs \N{GREEK CAPITAL LETTER DELTA} Accuracy (' + scenario + ')', fontsize=18)
    for key in data_sets.keys():
        data = data_sets[key]
        cond_list = []
        mins_list = []
        maxs_list = []
        accuracy_list = []
        for d in data:
            mins = min(d['stats'][layer_name]['singular_values'].numpy())
            maxs = max(d['stats'][layer_name]['singular_values'].numpy())
            cond = float(maxs)/float(mins)
            cond_list.append(cond)
            mins_list.append(mins)
            maxs_list.append(maxs)
            accuracy_list.append(d['curr_acc1'])

        diff_acc = [j-i for i, j in zip(accuracy_list[:-1], accuracy_list[1:])]
        diff_cond = [j-i for i, j in zip(cond_list[:-1], cond_list[1:])]

        # for i in range(0, len(cond_list)-1):
        plt.plot(diff_cond, diff_acc, marker=r'$\bowtie$', linestyle='None', label=key)
        plt.ylabel(u"\N{GREEK CAPITAL LETTER DELTA}  Accuracy %", fontsize=18)
        plt.xlabel(u"\N{GREEK CAPITAL LETTER DELTA}  Conditioning", fontsize=18)
        plt.legend(loc='lower right')
        # plt.xscale('symlog')
            # plt.quiver(cond_list[i], accuracy_list[i],  cond_list[i+1] - cond_list[i], accuracy_list[i+1]- accuracy_list[i], angles='xy', scale_units='xy', scale=1, label=key)



    plt.grid(True)
    # plt.xscale('symlog')
    plt.show(block=False)
    plt.legend(loc='lower right')

def box_plot_singular_values(data_sets, layer_name):

    # sucks this is copy pasted ):
    for key in data_sets.keys():
        fig, ax = plt.subplots()
        fig.suptitle(key + " Eigen Value Distributions vs training epochs", fontsize=18)
        data = data_sets[key]
        cond_list = []
        mins_list = []
        maxs_list = []
        accuracy_list = []
        for d in data:
            mins = min(d['stats'][layer_name]['singular_values'].numpy())
            maxs = max(d['stats'][layer_name]['singular_values'].numpy())
            cond = float(maxs)/float(mins)
            cond_list.append(cond)
            mins_list.append(mins)
            maxs_list.append(maxs)
            accuracy_list.append(d['curr_acc1'])
        
        ax.set_title("Box Plots of Singular Values")
        box = []
        for d in data:
            box.append(d['stats']['layer3.weight']['singular_values'].numpy())
        # ax.boxplot(box)
        # plt.show(block=False)
        # print(np.concatenate(box[0:9]))

        plt.subplot(4, 1, 1)
        plt.hist(box[0:9], 10)
        plt.ylim(0, 35)
        plt.xlim(0, 1.5)
        plt.xlabel('Eigen Values Epochs 0:9', fontsize=18)

        plt.subplot(4, 1, 2)
        plt.hist(box[10:19], 10)
        plt.ylim(0, 35)
        plt.xlim(0, 1.5)
        plt.xlabel('Eigen Values Epochs 9:19', fontsize=18)

        plt.subplot(4, 1, 3)
        plt.hist(box[20:29], 10)
        plt.ylim(0, 35)
        plt.xlim(0, 1.5)
        plt.xlabel('Eigen Values Epochs 20:29', fontsize=18)

        plt.subplot(4, 1, 4)
        plt.hist(box[30:39], 10)
        plt.ylim(0, 35)
        plt.xlim(0, 1.5)
        plt.xlabel('Eigen Values Epochs 20:29', fontsize=18)

    plt.show(block=False)


def plot_conditioning(data, layer_name, scenario):

    plot_singular_values_over_epochs(data, layer_name, scenario)
    plot_accuracy(data, layer_name, scenario)
    plot_accuracy_v2(data, layer_name, scenario)
    # plot_singular_values(data, layer_name)
    # box_plot_singular_values(data, layer_name)
    # fig = plt.figure()

    

    # plt.show(block=False)

    # fig = plt.figure()
    # plt.grid(True)
    # plt.xscale('symlog')
    # for i in range(0, len(cond_list)-1):
    #     plt.quiver(cond_list[i], accuracy_list[i],  cond_list[i+1] - cond_list[i], accuracy_list[i+1]- accuracy_list[i], angles='xy', scale_units='xy', scale=1)
    # plt.show(block=False)

def load_folder(folder_name):
    if os.path.isdir(folder_name):
            print("Loading folder contents %s", folder_name)
            models = listdir(folder_name)
            models.sort()
            i = 0
            data = []
            for m in models:
                # no point in wasting GPU resources on this....
                data.append(torch.load(folder_name + "/" + m, map_location=lambda storage, loc: storage))
                i = i + 1
            return data
            # analyze(data)
    else:
        print("No folder found at %s", folder_name)

def build_understanding(root_folder):
    data = {}
    if os.path.isdir(root_folder):
        print("Loading folder contents %s", root_folder)
        sub_models = listdir(root_folder)
        for s in sub_models:
            data[s] = load_folder(root_folder + "/" + s)
    else:
        print(root_folder +" is not a folder")

    return data


def main():
    args = parser.parse_args()
    data = build_understanding(args.dir)
    plot_conditioning(data, args.layer_name, args.scenario)
    print(data.keys())
    input("Press Enter to continue...")

if __name__ == '__main__':
    main()

