from __future__ import print_function
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

def std_normalize(data):
    normalized = data - np.mean(data)
    normalized /= np.std(data)
    return normalized


class LeNet(nn.Module):
    """LeNet CNN architecture."""
    def __init__(self):
        super(LeNet, self).__init__()
        self.conv1 = nn.Conv2d(1, 6, 3)
        self.conv2 = nn.Conv2d(6, 16, 3)
        self.drop1 = nn.Dropout(p=0.5)
        self.fc1 = nn.Linear(16 * 6 * 6, 120)  # 6*6 from image dimension
        self.drop2 = nn.Dropout(p=0.2)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 2)

    def forward(self, x):
        # Max pooling over a (2, 2) window
        x = F.max_pool2d(F.relu(self.conv1(x)), 2)
        x = F.max_pool2d(F.relu(self.conv2(x)), 2)
        x = self.drop1(x)
        x = x.view(-1, 16 * 6 * 6)
        x = F.relu(self.fc1(x))
        x = self.drop2(x)
        x = F.relu(self.fc2(x))
        x = F.softmax(self.fc3(x), dim=1)
        return x


class PoolLearner:
    """Pool-based active CNN learner that allows several common query strategies."""
    def __init__(self, total_pool, eta=0.001, epochs=10):
        total_pool = np.array(total_pool)
        assert total_pool.shape[1] == 32 and total_pool.shape[2] == 32
        self.total_pool = torch.from_numpy(std_normalize(total_pool)).reshape(
                              (total_pool.shape[0], 1, 32, 32)).float()

        self.X_labeled = []
        self.Y_labeled = []
        self.unlabeled_idxs = set(range(len(total_pool)))

        self.model = LeNet().float()
        self.epochs = epochs
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = optim.SGD(self.model.parameters(), lr=eta, momentum=0.9)

        self.device = torch.device("cpu")
        self.model = self.model.to(self.device)
        self.total_pool = self.total_pool.to(self.device)

    def give_label(self, i, label):
        assert i < len(self.total_pool)
        if i in self.unlabeled_idxs:
            self.X_labeled.append(self.total_pool[i])
            self.Y_labeled.append(torch.tensor(label).long().to(self.device))
            self.unlabeled_idxs.remove(i)

    def train(self, verbose=False):
        def weights_init(m):
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                m.bias.data.fill_(0.01)
        self.model.train()
        self.model.apply(weights_init)
        for epoch in range(self.epochs):
            running_loss = 0.
            shuffled_idxs = np.random.permutation(len(self.X_labeled))
            for i in range(len(self.X_labeled)):
                inputs, labels = self.X_labeled[shuffled_idxs[i]].reshape((1, 1, 32, 32)), \
                                 self.Y_labeled[shuffled_idxs[i]].reshape((1, ))

                # zero the parameter gradients
                self.optimizer.zero_grad()

                # forward + backward + optimize
                outputs = self.model(inputs)
                loss = self.criterion(outputs, labels)
                loss.backward()
                self.optimizer.step()

                running_loss += loss.item()
            if verbose:
                print('Training loss: %f' % running_loss)

    def lcus_query(self, penalties=None):
        with torch.no_grad():
            self.model.eval()
            outputs = self.model(self.total_pool)
            confidences, _ = torch.max(outputs, 1)
        query_idx = 0
        max_score = -float("inf")
        for i in self.unlabeled_idxs:
            score = -confidences[i]
            if penalties:
                score -= penalties[i]
            if score > max_score:
                max_score = score
                query_idx = i
        return query_idx

    def predict_on_pool(self):
        with torch.no_grad():
            self.model.eval()
            outputs = self.model(self.total_pool)
            return torch.max(outputs, 1)[1]
