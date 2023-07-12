import scipy.io
import numpy as np
from matplotlib import pyplot as plt
from sklearn.cluster import KMeans
from scipy.io import savemat


# implement kmeans on a full-sized picture
full_sized_pic = np.array(scipy.io.loadmat('name'))
X = []
for i in range(full_sized_pic.shape[0]):
    for j in range(full_sized_pic.shape[1]):
        X.append(full_sized_pic[i, j, :])
kmeans = KMeans(n_clusters=5, random_state=0).fit(X)
ClusteredPixel = np.zeros((full_sized_pic.shape[0], full_sized_pic.shape[1]), dtype=int)
for x in range(0, full_sized_pic.shape[0]):
    for y in range(0, full_sized_pic.shape[1]):
        ClusteredPixel[x, y] = kmeans.predict(full_sized_pic[x, y, :].reshape(1, -1))
plt.figure(figsize=(10, 6))
plt.imshow(ClusteredPixel, interpolation='nearest')
plt.title('kmeans on original pic')
plt.savefig('kmeans on original pic')

# slice the full-sized picture to small pictures
def split_photo(base_name,matrix, num_of_horizontal_cuts, num_of_vertical_cuts):
    for i in range(num_of_horizontal_cuts):
        for j in range(num_of_vertical_cuts):
            dict_to_save = {base_name + str((j * num_of_horizontal_cuts) + i ): matrix[i * round(matrix.shape[0]/num_of_vertical_cuts):
                                                                                       (i + 1) *round(matrix.shape[0]/num_of_vertical_cuts) ,
                                                                                j * round(matrix.shape[0]/num_of_horizontal_cuts) :
                                                                                (j + 1) * round(matrix.shape[0]/num_of_horizontal_cuts) , :],
                     "label": "pic"}
            savemat(base_name + str((j * num_of_horizontal_cuts) + i ) + ".mat", dict_to_save)

# concatenate pictures to create a small train_set set for verification
train_set = np.array(scipy.io.loadmat('/content/pictnum0.mat')['pictnum0'])
for i in range(1,4):
    train_set = np.concatenate((train_set, np.array(scipy.io.loadmat(f'/content/pictnum{i}.mat')[f'pictnum{i}'])), axis = 0)
X = []
for i in range(train_set.shape[0]):
    for j in range(train_set.shape[1]):
        X.append(train_set[i, j, :])
kmeans = KMeans(n_clusters=5, random_state=0).fit(X)
ClusteredPixel = np.zeros((train_set.shape[0], train_set.shape[1]), dtype=int)
for x in range(0, train_set.shape[0]):
    for y in range(0, train_set.shape[1]):
        ClusteredPixel[x, y] = kmeans.predict(train_set[x, y, :].reshape(1, -1))
plt.figure(figsize=(20, 10))
plt.imshow(ClusteredPixel, interpolation='nearest')
plt.title('kmeans on train_set set')
plt.savefig('kmeans on train_set set')

# compare the small train_set set to the same area
# trained as a full-sized picture
plt.figure(figsize=(10, 6))
plt.imshow(ClusteredPixel[0:548,0:55], interpolation='nearest')
plt.title('kmeans slice on original set')
plt.savefig('kmeans slice on original set')

# train_set a set, using 80-20 train_set and test sets
train_set = np.array(scipy.io.loadmat('/content/pictnum0.mat')['pictnum0'])
for i in range(1,80):
    train_set = np.concatenate((train_set, np.array(scipy.io.loadmat(f'/content/pictnum{i}.mat')[f'pictnum{i}'])), axis = 0)
X = []
for i in range(train_set.shape[0]):
    for j in range(train_set.shape[1]):
        X.append(train_set[i, j, :])
kmeans = KMeans(n_clusters=5, random_state=0).fit(X)
ClusteredPixel = np.zeros((train_set.shape[0], train_set.shape[1]), dtype=int)
for x in range(0, train_set.shape[0]):
    for y in range(0, train_set.shape[1]):
        ClusteredPixel[x, y] = kmeans.predict(train_set[x, y, :].reshape(1, -1))
Y = kmeans.labels_

def L2(v1, v2):
  return np.linalg.norm(v1-v2)

# predict a label for a new picture using the trained labels
unlabeled_im = np.array(scipy.io.loadmat('/content/pictnum85.mat')['pictnum85'])
minCluster = np.zeros((unlabeled_im.shape[0], unlabeled_im.shape[1]), dtype=int)
centroids = np.array(kmeans.cluster_centers_)
for x in range(unlabeled_im.shape[0]):
    for y in range(unlabeled_im.shape[1]):
        min = float('inf')
        for cluster in range(len(kmeans.cluster_centers_)):
            if (L2(unlabeled_im[x, y, :], centroids[cluster]) < min):
                minCluster[x, y] = cluster
                min = L2(unlabeled_im[x, y, :], centroids[cluster])

fig, ax = plt.subplots(1, 2, figsize=(10, 10))
ax[0].imshow(minCluster, interpolation='nearest')
ax[0].set_title('generated labels')
ax[1].imshow(unlabeled_im[:, :, 30], interpolation='nearest')
ax[1].set_title('original pic')
fig.savefig('generated labels')

# predict a label for a new picture using the trained labels
# and identify anomaly detection
defected_image = np.array(scipy.io.loadmat('/content/falsed_pic.mat')['falsed_pic'])
minCluster = np.zeros((defected_image.shape[0], defected_image.shape[1]), dtype=int)
centroids = np.array(kmeans.cluster_centers_)
for x in range(defected_image.shape[0]):
    for y in range(defected_image.shape[1]):
        min = float('inf')
        for l in range(len(kmeans.cluster_centers_)):
            if (L2(defected_image[x, y, :], centroids[l]) < min):
                minCluster[x, y] = l
                min = L2(defected_image[x, y, :], centroids[l])

fig, ax = plt.subplots(1, 3, figsize=(10, 10))
ax[0].imshow(minCluster, interpolation='nearest')
ax[0].set_title('generated labels')
ax[1].imshow(defected_image[:, :, 30], interpolation='nearest')
ax[1].set_title('original pic')

X = []
for i in range(defected_image.shape[0]):
    for j in range(defected_image.shape[1]):
        X.append(defected_image[i, j, :])
kmeans = KMeans(n_clusters=5, random_state=0).fit(X)
ClusteredPixel = np.zeros((defected_image.shape[0], defected_image.shape[1]), dtype=int)
for x in range(0, defected_image.shape[0]):
    for y in range(0, defected_image.shape[1]):
        ClusteredPixel[x, y] = kmeans.predict(defected_image[x, y, :].reshape(1, -1))

ax[2].imshow(ClusteredPixel, interpolation='nearest')
ax[2].set_title('kmeans')
fig.savefig('comparison between algorithms defected')

# plot a histogram of the distances of each pixel from the k-clusters
defected_image = np.array(scipy.io.loadmat('/content/falsed_pic.mat')['falsed_pic'])
minCluster = np.zeros((defected_image.shape[0], defected_image.shape[1]), dtype=int)
centroids = np.array(kmeans.cluster_centers_)
distances = []
for x in range(defected_image.shape[0]):
    for y in range(defected_image.shape[1]):
        min = float('inf')
        for cluster in range(len(kmeans.cluster_centers_)):
            if (L2(defected_image[x,y,:], centroids[cluster]) < min):
                minCluster[x,y] = cluster
                min = L2(defected_image[x,y,:], centroids[cluster])
                distances.append(min)

plt.figure(figsize=(10, 7))
hist, bins = np.histogram(distances, bins = range(102))
plt.hist(distances)
plt.title('histogram of pic with artificial anomaly')
plt.savefig('histogram of pic with artificial anomaly')
