#!/home/azhai/anaconda3/envs/py27/bin/python2.7
import numpy as np
import cv2
import matplotlib.pyplot as plt
from actvlrn import PoolLearner
from sklearn.cluster import KMeans

TRUE_SAFE_MAP_FILE = '/home/azhai/catkin_ws/src/safe_mapping/resource/easy_tables_true_map.png'
HIGH_VIEW_FILE = '/home/azhai/catkin_ws/src/safe_mapping/resource/easy_tables_high_view.jpg'

def load_true_map(imgfile):
    return cv2.imread(imgfile, 0)

def query_true_map(true_map, x, y):
    x_idx = int(x * 100)
    y_idx = int(y * 100)
    if true_map[x_idx][y_idx] == 0:
        return 0
    else:
        return 1

def get_pool_labels(positions, true_map):
    output = []
    for i in range(len(positions)):
        x, y = positions[i]
        output.append(query_true_map(true_map, x, y))
    return output

def get_avg_intensities(pool):
    avg_intensities = []
    for i in range(len(pool)):
        avg_intensities.append([np.mean(pool[i])])
    return avg_intensities

def kmeans_warm_start(avg_intensities, K=10):
    kmeans = KMeans(n_clusters=K, random_state=0).fit(avg_intensities)
    warm_start = []
    for i in range(K):
        best_j = 0
        min_dist = float('inf')
        for j in range(len(avg_intensities)):
            dist = abs(kmeans.cluster_centers_[i] - avg_intensities[j])
            if dist < min_dist:
                min_dist = dist
                best_j = j
        warm_start.append(best_j)
    return warm_start

def make_grid_pool(img, grid_size, img_scale):
    pool, positions = [], []
    x_min = (img.shape[0] % grid_size) / 2
    while x_min + grid_size <= img.shape[0]:
        y_min = (img.shape[1] % grid_size) / 2
        while y_min + grid_size <= img.shape[1]:
            pool.append(img[x_min:(x_min + grid_size), y_min:(y_min + grid_size)])
            positions.append(((x_min + grid_size / 2.) / img_scale, (y_min + grid_size / 2.) / img_scale))
            y_min += grid_size
        x_min += grid_size
    return pool, positions

def make_dense_pool(high_view, img_size, img_scale):
    pool, positions = [], []
    for x_min in range(high_view.shape[0] - 32):
        for y_min in range(high_view.shape[1] - 32):
            pool.append(high_view[x_min:(x_min + img_size), y_min:(y_min + img_size)])
            positions.append(((x_min + img_size / 2.) / img_scale, (y_min + img_size / 2.) / img_scale))
    return pool, positions

def query_label_retrain(learner, pool_labels):
    query_idx = learner.lcus_query()
    learner.give_label(query_idx, pool_labels[query_idx])
    learner.train()

def pool_accuracy(learner, pool_labels):
    preds = learner.predict_on_pool()
    count, total = 0, 0
    for i in learner.unlabeled_idxs:
        if preds[i] == pool_labels[i]:
            count += 1
        total += 1
    return float(count) / total

def train_warm_start(learner, pool_labels, warm_start, verbose=False):
    for i in warm_start:
        learner.give_label(i, pool_labels[i])
    learner.train(verbose=verbose)

def train_entire_pool(learner, pool_labels, verbose=False):
    for i in range(len(pool_labels)):
        learner.give_label(i, pool_labels[i])
    learner.train(verbose=verbose)

def simulate_random_sampling(learner, pool_labels, queries, verbose=False):
    history = np.zeros(queries)
    rand_sample = np.random.choice(range(len(pool)), queries, replace=False)
    for i in range(queries):
        learner.give_label(rand_sample[i], pool_labels[rand_sample[i]])
        learner.train()
        pool_acc = pool_accuracy(learner, pool_labels)
        if verbose:
            print('-' * 80)
            print('Queries made: %d\nAccuracy on unlabeled pool samples: %f' % (i + 1, pool_acc))
        history[i] = pool_acc
    return history

def simulate_active_learning(learner, pool_labels, queries, verbose=False):
    history = np.zeros(queries)
    for i in range(queries):
        query_label_retrain(learner, pool_labels)
        pool_acc = pool_accuracy(learner, pool_labels)
        if verbose:
            print('-' * 80)
            print('Queries made: %d\nAccuracy on unlabeled pool samples: %f' % (i + 1, pool_acc))
        history[i] = pool_acc
    return history

def plot_learning_curve(avg_history, std_history, title, outfile):
    xs = range(1, len(avg_history) + 1)
    plt.plot(xs, avg_history)
    plt.fill_between(xs, avg_history - std_history, avg_history + std_history, alpha=0.2)
    plt.xlabel('# of queries made')
    plt.ylabel('Accuracy on unlabeled pool samples')
    plt.title(title)
    plt.savefig(outfile)
    plt.close()

if __name__ == '__main__':
    true_map = load_true_map(TRUE_SAFE_MAP_FILE)
    high_view = cv2.imread(HIGH_VIEW_FILE, 0)
    img_scale = high_view.shape[0] / 5.
    pool, positions = make_dense_pool(high_view, 32, img_scale)
    pool_labels = get_pool_labels(positions, true_map)
    print('Pool size: %d' % len(pool))

    # accs = []
    # for _ in range(30):
    #     rand_sample = np.random.choice(range(len(pool)), 100, replace=False)
    #     learner = PoolLearner(pool)
    #     train_warm_start(learner, pool_labels, rand_sample)
    #     accs.append(pool_accuracy(learner, pool_labels))
    #     print(accs[-1])
    # print(np.mean(accs))

    # QUERIES = 200
    # REPS = 10
    # histories = np.zeros((REPS, QUERIES))
    # for rep in range(REPS):
    #     learner = PoolLearner(pool)
    #     histories[rep] = simulate_active_learning(learner, pool_labels, QUERIES, verbose=True)
    #     print('%d repetitions simulated...' % (rep + 1))
    # avg_history = np.mean(histories, axis=0)
    # std_history = np.std(histories, axis=0)
    # plot_learning_curve(avg_history, std_history,
    #                     'Least-Confidence Uncertainty Sampling (Cold Start)',
    #                     '/home/azhai/Documents/safe_mapping/figures/lcus_cold10.png')
    # np.save('/home/azhai/Documents/safe_mapping/histories/lcus_cold10.npy', histories)

    QUERIES = 200
    REPS = 10
    histories = np.zeros((REPS, QUERIES))
    for rep in range(REPS):
        learner = PoolLearner(pool)
        histories[rep] = simulate_random_sampling(learner, pool_labels, QUERIES, verbose=True)
        print('%d repetitions simulated...' % (rep + 1))
    avg_history = np.mean(histories, axis=0)
    std_history = np.std(histories, axis=0)
    plot_learning_curve(avg_history, std_history,
                        'Uniform Random Sampling',
                        '/home/azhai/Documents/safe_mapping/figures/urs10.png')
    np.save('/home/azhai/Documents/safe_mapping/histories/urs10.npy', histories)

