import numpy as np
import json
import matplotlib.pyplot as plt

PATH_TO_DATA = '/home/azhai/Documents/safe_mapping/histories/'
PATH_TO_FIGURES = '/home/azhai/Documents/safe_mapping/figures/'

def plot_learning_curve(avg_history, std_history, title, outfile):
    xs = range(1, len(avg_history) + 1)
    plt.plot(xs, avg_history)
    plt.fill_between(xs, avg_history - std_history, avg_history + std_history, alpha=0.2)
    plt.xlabel('# of queries made')
    plt.ylabel('Accuracy on unlabeled pool samples')
    plt.title(title)
    plt.savefig(outfile)
    plt.close()

def plot_learning_curves(avgs, stds, names, outfile, bin_size=1):
    colors = ['r', 'g', 'b', 'c', 'm', 'y', 'k']
    xs = range(bin_size, (len(avgs[0]) + 1) * bin_size, bin_size)
    for i in range(len(avgs)):
        plt.plot(xs, avgs[i], label=names[i], color=colors[i])
        plt.fill_between(xs, avgs[i] - stds[i], avgs[i] + stds[i], color=colors[i], linewidth=0, alpha=0.2)
    plt.xlabel('# of queries made')
    plt.ylabel('Accuracy on unlabeled pool samples')
    plt.title('Comparison of Active Query Strategies')
    plt.legend(loc='best')
    plt.savefig(PATH_TO_FIGURES + outfile)
    plt.close()

def add_new_strategy(datafile, name, avgs, stds, names, bin_size=1):
    histories = np.load(PATH_TO_DATA + datafile + '.npy')
    bins = range(0, histories.shape[1], bin_size)
    avg_history = np.zeros(len(bins))
    std_history = np.zeros(len(bins))
    for i, bin in enumerate(bins):
        avg_history[i] = np.mean(histories[:, bin:bin+bin_size])
        std_history[i] = np.std(histories[:, bin:bin+bin_size])
    avgs.append(avg_history)
    stds.append(std_history)
    names.append(name)

def plot_reward_history(result_file, title, outfile):
    hist = []
    with open(result_file) as f:
        for line in f:
            data = json.loads(line)  # JSON dictionary
            hist.append(data['episode_reward_mean'])
    plt.plot(range(1, len(hist) + 1), hist)
    plt.xlabel('# of Episodes')
    plt.ylabel('Episode Reward')
    plt.title(title)
    plt.savefig(outfile)
    plt.close()


if __name__ == '__main__':
    avgs, stds, names = [], [], []
    bin_size = 1
    add_new_strategy('lcus_cold10', 'Least-Confidence Uncertainty', avgs, stds, names, bin_size=bin_size)
    add_new_strategy('urs10', 'Uniform Random', avgs, stds, names, bin_size=bin_size)
    plot_learning_curves(avgs, stds, names, 'lcus_urs.png', bin_size=bin_size)

