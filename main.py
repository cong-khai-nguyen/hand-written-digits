from sklearn.datasets import load_digits
from sklearn.cluster import KMeans
from sklearn.preprocessing import scale
import numpy as np
import pylab as pl
from sklearn import metrics

digits = load_digits()
data = scale(digits.data) #features
print(digits.images)

y = digits.target # target

# In this case 1797 samples and 64 features
samples, features = data.shape

print(samples, features)


def bench_k_means(estimator, name, data):
    estimator.fit(data)
    print('%-9s\t%i\t%.3f\t%.3f\t%.3f\t%.3f\t%.3f\t%.3f'
          % (name, estimator.inertia_,
             metrics.homogeneity_score(y, estimator.labels_),
             metrics.completeness_score(y, estimator.labels_),
             metrics.v_measure_score(y, estimator.labels_),
             metrics.adjusted_rand_score(y, estimator.labels_),
             metrics.adjusted_mutual_info_score(y,  estimator.labels_),
             metrics.silhouette_score(data, estimator.labels_,
                                      metric='euclidean')))
# Find out the number of centroids to use K-Means Clustering
k = len(np.unique(y)) #10

clf = KMeans(n_clusters= k, init = "random", n_init= 10)
bench_k_means(clf, "Score", data)

pl.gray()
pl.matshow(digits.images[0])
pl.show()