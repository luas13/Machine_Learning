import numpy as np
import matplotlib.pyplot as plt
import math


class kMeans(object):
    def __init__(self):
        self.k = 3
        self.centroids = []
        self.points = []
        self.dim = 0

    def initialize_dataset(self):
        self.points = np.loadtxt("seed_dataset.txt", unpack=False)
        self.dim = len(self.points[0])

    def initialize_centroid(self):
        self.centroids = self.points.copy()
        np.random.shuffle(self.points)
        self.centroids = self.centroids[:self.k]
        return self.centroids

    def closest_centroid(self):
        distances = np.sqrt(((self.points - self.centroids[:, np.newaxis]) ** 2).sum(axis=2))
        return np.argmin(distances, axis=0)

    def move_centroids(self, closest):
        """returns the new centroids assigned from the points closest to them"""
        return np.array([self.points[closest == c].mean(axis=0) for c in range(self.centroids.shape[0])])


def converged(centroids, old_centroids):
    if len(old_centroids) == 0:
        return False

    if len(centroids) <= 5:
        a = 1
    elif len(centroids) <= 10:
        a = 1
    else:
        a = 4

    for i in range(0, len(centroids)):
        cent = centroids[i]
        old_cent = old_centroids[i]

        if ((int(old_cent[0]) - a) <= cent[0] <= (int(old_cent[0]) + a)) and ((int(old_cent[1]) - a) <= cent[1] <= (int(old_cent[1]) + a))\
                and ((int(old_cent[2]) - a) <= cent[2] <= (int(old_cent[2]) + a)) and ((int(old_cent[3]) - a) <= cent[3] <= (int(old_cent[3]) + a)) \
                and ((int(old_cent[4]) - a) <= cent[4] <= (int(old_cent[4]) + a)) and ((int(old_cent[5]) - a) <= cent[5] <= (int(old_cent[5]) + a)) \
                and ((int(old_cent[6]) - a) <= cent[6] <= (int(old_cent[6]) + a)):
            continue
        else:
            return False
    return True


if __name__ == '__main__':
    obj = kMeans()
    obj.initialize_dataset()

    k_array = [2, 3, 4, 5, 6, 7, 8, 9, 10]
    loss_dict = {}
    for i in range(obj.k):
        loss_dict[i] = 0

    for k in range(2, 11):
        obj.k = k
        old_centroid = []
        new_centroids = obj.initialize_centroid()

        # plt.scatter(obj.points[:, 0], obj.points[:, 1])
        # plt.scatter(new_centroids[:, 0], new_centroids[:, 1], c='r', s=100)
        # plt.show()

        i = 0
        while not converged(new_centroids, old_centroid) and i < 30:
            closest = obj.closest_centroid()
            old_centroid = new_centroids
            new_centroids = obj.move_centroids(closest)
            i += 1

            # plt.subplot(121)
            # plt.scatter(obj.points[:, 0], obj.points[:, 1])
            # plt.scatter(old_centroid[:, 0], old_centroid[:, 1], c='r', s=100)
            #
            # plt.subplot(122)
            # plt.scatter(obj.points[:, 0], obj.points[:, 1])
            # plt.scatter(new_centroids[:, 0], new_centroids[:, 1], c='r', s=100)

            # plt.show()

        print "Convergence reached!!!"

        key_arr = []
        for i in range(obj.k):
            key_arr.append(i)
        mycentroid = {}

        print "len(mycentroid)", len(new_centroids)
        for i in range(len(new_centroids)):
            mycentroid[i] = 0

        for i in range(len(obj.points)):
            index = closest[i]
            x = obj.points[i]
            y = new_centroids[index]

            ed = sum((x-y) ** 2)
            mycentroid[index] += ed

        loss_func = 0
        for i in range(len(new_centroids)):
            print "i: ", i, "mycentroid[i]", mycentroid[i]
            loss_func += mycentroid[i]
        loss_dict[k] = loss_func

    train_lists = sorted(loss_dict.items())
    k, e = zip(*train_lists)
    plt1 = plt.subplot(111)
    plt1.set_xlim(2, 10)
    plt1.plot(k, e, 'r-o', label='Objective function w.r.t. k')

    # k, e1 = zip(*train_lists)
    # e = [x/7 for x in e1]
    # plt1 = plt.subplot(122)
    # plt1.set_xlim(2, 11)
    # plt1.plot(k, e, 'r-o', label='Loss function w.r.t. k')

    plt.legend(loc=5)

    plt.title("Objective function vs. the number of clusters k", weight='bold')
    plt.xlabel("k (no. of centroids)", weight='bold')
    plt.ylabel("Objective function", weight='bold')

    plt.show()
