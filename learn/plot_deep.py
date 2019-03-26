# https://matplotlib.org/users/legend_guide.html
import numpy as np
import matplotlib.pyplot as plt 


#################
################# PLOTTING
#################

class plot_deep:
    
    def plot_line(value_history):
        return plot_labeled_line(value_history, '', 'input(t)', 'output($)') 

    def plot_labeled_line(value_history, overall_title, x_label, y_label):
        time_ticks = np.array([column[0] for column in value_history])
        account_ticks = np.array([column[1] for column in value_history])
        plt.xlabel(x_label)
        plt.ylabel(y_label)
        balance_history_plot = plt.plot(time_ticks, account_ticks)  # fig, axs = 
        plt.title(overall_title)
        return balance_history_plot 

    def plot_multi_line(value_history_1, label_1, value_history_2, label_2,
                        overall_title, x_label, y_label):
        time_ticks_1 = np.array([column[0] for column in value_history_1])
        account_ticks_1 = np.array([column[1] for column in value_history_1])
        time_ticks_2 = np.array([column[0] for column in value_history_2])
        account_ticks_2 = np.array([column[1] for column in value_history_2])

        line_1, = plt.plot(time_ticks_1, account_ticks_1, label=label_1)   
        line_2, = plt.plot(time_ticks_2, account_ticks_2, label=label_2)   
        plt.legend(handles=[line_1, line_2], loc=2)

        plt.title(overall_title)
        plt.xlabel(x_label)
        plt.ylabel(y_label)
        
        
    def box_plot_singular_values(data):
        fig, ax = plt.subplots()
        ax.set_title("Box Plots of Singular Values")
        box = []
        for d in data:
           box.append(d['stats']['layer3.weight']['singular_values'].numpy())
        ax.boxplot(box)
        plt.show(block=False)

        fig = plt.figure()
        plt.title('Eigen Value Distributions vs training epochs')
        plt.subplot(4, 1, 1)
        plt.hist(box[0:9])
        plt.xlabel('Eigen Values Epochs 0:9')

        plt.subplot(4, 1, 2)
        plt.hist(box[10:19])
        plt.xlabel('Eigen Values Epochs 9:19')

        plt.subplot(4, 1, 3)
        plt.hist(box[20:29])
        plt.xlabel('Eigen Values Epochs 20:29')

        plt.subplot(4, 1, 4)
        plt.hist(box[30:39])
        plt.show(block=False)
        plt.xlabel('Eigen Values Epochs 20:29')

    def conditioning(data):
        cond_list = []
        mins_list = []
        maxs_list = []
        accuracy_list = []
        fig = plt.figure()
        for d in data:
           mins = min(d['stats']['layer3.weight']['singular_values'].numpy())
           maxs = max(d['stats']['layer3.weight']['singular_values'].numpy())
           cond = float(maxs)/float(mins)
           cond_list.append(cond)
           mins_list.append(mins)
           maxs_list.append(maxs)
           accuracy_list.append(d['curr_acc1'])

        plt.title('Singular Values vs Epochs')
        plt.subplot(3, 1, 1)
        plt.grid(True)
        plt.xlabel('Epoch #')
        plt.ylabel('Minimum singular value')
        plt.ylim(0, 0.005)
        plt.plot(mins_list, linestyle='None', marker=r'$\bowtie$')

        plt.subplot(3, 1, 2)
        plt.grid(True)
        plt.xlabel('Epoch #')
        plt.ylabel('Maximum singular value')
        plt.ylim(0, 1.5)
        plt.plot(maxs_list, linestyle='None', marker=r'$\bowtie$')

        plt.subplot(3, 1, 3)
        plt.grid(True)
        plt.xlabel('Epoch #')
        plt.ylabel('Conditioning')
        plt.yscale('symlog')
        # plt.ylim(10, 10e9)
        plt.plot(cond_list, linestyle='None', marker=r'$\bowtie$')


        fig = plt.figure()

        diff_acc = [j-i for i, j in zip(accuracy_list[:-1], accuracy_list[1:])]

        plt.subplot(3, 1, 1)
        plt.grid(True)
        plt.xlabel('Epoch #')
        plt.ylabel('Accuracy')
        plt.ylim(0, 15)
        plt.plot(accuracy_list, linestyle='None', marker=r'$\bowtie$')

        plt.subplot(3, 1, 2)
        plt.grid(True)
        plt.xlabel('Epoch #')
        plt.ylabel('Diff Accuracy')
        plt.ylim(-1.5, 1.5)
        plt.plot(diff_acc, linestyle='None', marker=r'$\bowtie$')

        plt.subplot(3, 1, 3)
        plt.grid(True)
        plt.xlabel('Epoch #')
        plt.ylabel('Conditioning')
        # plt.ylim(10, 10e5)
        plt.plot(cond_list, linestyle='None', marker=r'$\bowtie$')

        plt.show(block=False)

        fig = plt.figure()
        plt.grid(True)
        plt.xscale('symlog')
        for i in range(0, len(cond_list)-1):
           plt.quiver(cond_list[i], accuracy_list[i],  cond_list[i+1] - cond_list[i], accuracy_list[i+1]- accuracy_list[i], angles='xy', scale_units='xy', scale=1)
        plt.show(block=False)
