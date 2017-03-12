import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import style
style.use('ggplot')

#reading the file
df = pd.read_csv('exercise-6.csv',delimiter=',',skipinitialspace=True)
# making a array list to hold the x values (first column) from the csv file
x = []
y = []
x.extend(np.around(df['x'],decimals=1)) # assigning the first column values to the array
y.extend(np.around(df['y'],decimals=1)) # assigning the second column values to the array
# formatting the array so every element in it is to 2 decimal places
formattedX=['%.2f' % elem for elem in x]
formattedY=['%.2f' % elem for elem in y]

# the matrix is the new array that we pass to the k-means class
newX = df.as_matrix()

plt.scatter(formattedX,formattedY)
plt.title('Original Unclustered Data')
plt.xlabel('x')
plt.ylabel('y')
plt.show()

colors = 10*["g","r","c","b","k"]

class K_Means:
    # toleraence is the percentage of how much the centroid is going to move, max_iter is the max iterations the program will make
    def __init__(self, k = 3, tolerance = 0.001, max_iter = 300):
        self.k = k
        self.tolerance = tolerance
        self.max_iter = max_iter

        #keeps a constant number of centroids to move
    def fit(self, data):
        self.centroids = {} #empty dictionary to store centroids

        #iterating through data, takes the first two data points as centroids to use
        for i in range(self.k):
            self.centroids[i] = data[i]

        #empties the centroids every time the centroid moves, so empty for every iteration
        for i in range(self.max_iter):
            self.classifications = {} #empty dictionary to store new centroids

            #iterating through centroids keys to equal empty list
            for i in range (self.k):
                self.classifications[i] = [] #empty list of feature set values

            #populating the empty list created above/ giving it values
            for featureset in data:
                # calculates the distance between the data and the centroid
                distances = [np.linalg.norm(featureset - self.centroids[centroid]) for centroid in self.centroids] #creating a list with k # of centroids
                classification = distances.index(min(distances)) #gets the min distance centroid and assign it to list.
                self.classifications[classification].append(featureset) #adds the new centroid found to the dictionary

            # comparing the two centroids, to see how much they have changed
            prev_centroids = dict(self.centroids)

            for classification in self.classifications:
                # takes the mean of each cluster to calculate new centroid of the cluster
                self.centroids[classification] = np.average(self.classifications[classification], axis = 0)

            optimized = True

            # if any of the centroids in their movement move more than the tolerance (0.001), we say we are not optimized
            for c in self.centroids:
                original_centroid = prev_centroids[c]
                current_centroid = self.centroids[c]

                if np.sum((current_centroid - original_centroid) / original_centroid * 100.0) > self.tolerance:
                    optimized = False
                # this will break the for loop and stop us from running every single maximum iteration
                # If this is not the case it will continue through the for loop until it reaches the maximum iteration and at that time whatever the centroid is. This is our final centroid
                if optimized:
                    break

clf = K_Means()
clf.fit(newX)
# plots the centroids
for centroid in clf.centroids:
    plt.scatter(clf.centroids[centroid][0] , clf.centroids[centroid][1], marker = "*", color = "k", s = 150)

# Classifies which color to use for each clusters
for classification in clf.classifications:
    color = colors[classification]

    for featureset in clf.classifications[classification]:
        plt.scatter(featureset[0], featureset[1], marker = "o", color = color, linewidths = 2)

plt.title('Clustered data')
plt.xlabel('x')
plt.ylabel('y')
plt.show()