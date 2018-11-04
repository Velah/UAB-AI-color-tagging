
"""
@author: Xavier Velasco Llaurado
"""
import numpy as np
import matplotlib.pyplot as plt
import mpl_toolkits.mplot3d.axes3d as axes3d
from sklearn.decomposition import PCA
import time

    
def distance(X,C):
    """@brief   Calculates the distance between each pixcel and each centroid 

    @param  X  numpy array PxD 1st set of data points (usually data points)
    @param  C  numpy array KxD 2nd set of data points (usually cluster centroids points)

    @return dist: PxK numpy array position ij is the distance between the 
        i-th point of the first set an the j-th point of the second set
    """
    if type(X) is not np.ndarray:
        X = np.array(X)
    if type(C) is not np.ndarray:
        C = np.array(C)
    X = np.reshape(X, (-1, X.shape[-1]))
    L = np.zeros((X.shape[0], C.shape[0]))

    for k in range(C.shape[0]):
        L[:,k] = np.sqrt(np.sum(np.square(np.subtract(X,C[k])), axis=1))
        
        
    return L
    

class KMeans():
    
    def __init__(self, X, K, options=None):
        """@brief   Constructor of KMeans class
        
        @param  X   LIST    input data
        @param  K   INT     number of centroids
        @param  options DICT dctionary with options
        """

        self._init_X(X)                                    # LIST data coordinates
        self._init_options(options)                        # DICT options
        self._init_rest(K)                                 # Initializes de rest of the object

        
    def _init_X(self, X):
        """@brief Initialization of all pixels
        
        @param  X   MATRIX  list of all pixel values. Usually it will be a numpy 
                            array containing an image NxMx3

        sets X an as an array of data in vector form (PxD  where P=N*M and D=3 in the above example)
        """
        self.X = np.array(np.reshape(X, (-1, X.shape[-1])))

            
    def _init_options(self, options):
        """@brief Initialization of options in case some fields are left undefined
        
        @param  options DICT dctionary with options

            sets de options parameters
        """
        if options == None:
            options = {}
        if not 'km_init' in options:
            options['km_init'] = 'first'
        if not 'verbose' in options:
            options['verbose'] = False
        if not 'tolerance' in options:
            options['tolerance'] = 0
        if not 'max_iter' in options:
            options['max_iter'] = np.inf
        if not 'fitting' in options:
            options['fitting'] = 'Fisher'

        self.options = options

        
    def _init_rest(self, K):
        """@brief   Initialization of the remainig data in the class.
        
        @param  options DICT dctionary with options
        """
        #print "INIT REST"
        self.K = K                                             # INT number of clusters
        if self.K>1:
            self._init_centroids()                             # LIST centroids coordinates
            self.old_centroids = np.empty_like(self.centroids) # LIST coordinates of centroids from previous iteration
            self.clusters = np.zeros(len(self.X))              # LIST list that assignes each element of X into a cluster
            self._cluster_points()                             # sets the first cluster assignation
        else:
            self.bestK()
            self._init_rest(self.K)                             
        self.num_iter = 0                                      # INT current iteration


    def _init_centroids(self):
        """@brief Initialization of centroids
        depends on self.options['km_init']
        """
        self.centroids = []
        if self.options['km_init'].lower() == 'first':
            # Estem agafant els indexs dels primers K punts de diferent color i 
            # utilitzant-los per a agafar aquests mateixos punts de la matriu X
            # Ho fem aixi perque np.unique retorna els valors ordenats i no ho volem
            # Utilitzem el np.sort perque np.unique no sempre retorna els index en ordre
            centroidIndexes = np.sort(np.unique(np.around(self.X, decimals = 0), axis = 0, return_index = True)[1])[:self.K]
            if len(centroidIndexes) < self.K:
                nonUniqueIndex = 0
                while len(centroidIndexes) < self.K:
                    if nonUniqueIndex not in centroidIndexes:
                        centroidIndexes = np.append(centroidIndexes, nonUniqueIndex)
                    nonUniqueIndex += 1
            self.centroids = self.X[centroidIndexes]
                
            
        if self.options['km_init'].lower() == 'random':
            self.centroids = self.X[np.random.random_integers(0, self.X.shape[0]-1,self.K)]
        
        if self.options['km_init'].lower() == 'diagonal':
            indexes = []
            for k in range(1,self.K+1):
                indexes.append(int(len(self.X)*(float(k)/(self.K+1))))
            self.centroids = self.X[indexes]
        
        np.nan_to_num(self.centroids, copy=False)
        if len(self.centroids) != self.K:
            print "CENTROIDS < K"
        
        
    def _cluster_points(self):
        """@brief   Calculates the closest centroid of all points in X
        """
        # Calculem la distancia de cada pixel a cada un dels centroids i obtenim
        # una matriu on cada fila conte la distancia d'aquell pixel a cada centroid
        dist = distance(self.X, self.centroids)
        # De cada fila n'agafem l'index del valor minimm (que es el numero de cluster)
        self.clusters = np.argmin(dist, axis=1)
        
    def _get_centroids(self):
        """@brief   Calculates coordinates of centroids based on the coordinates 
                    of all the points assigned to the centroid
        """
        self.old_centroids = np.copy(self.centroids)
        for k in range(len(self.centroids)):
            self.centroids[k] = np.mean(self.X[self.clusters==k], axis=0)
        np.nan_to_num(self.centroids, copy=False)
        
        if len(self.centroids) != self.K:
            if len(self.centroids) < self.K:
                print "CENTROIDS < K"
            else:
                print "CENTROIDS > K"
                           

    def _converges(self):
        """@brief   Checks if there is a difference between current and old centroids
        """
        return np.allclose(self.centroids, self.old_centroids, atol=self.options['tolerance'], rtol=0)
        
    def _iterate(self, show_first_time=True):
        """@brief   One iteration of K-Means algorithm. This method should 
                    reassigne all the points from X to their closest centroids
                    and based on that, calculate the new position of centroids.
        """
        self.num_iter += 1
        self._cluster_points()
        self._get_centroids()
        if self.options['verbose']:
            self.plot(show_first_time)


    def run(self):
        """@brief   Runs K-Means algorithm until it converges or until the number
                    of iterations is smaller than the maximum number of iterations.=
        """
        if self.K < 2:
            self.bestK()
            return        
        
        self._iterate(True)
        while not self._converges() and self.options['max_iter'] > self.num_iter:
            self._iterate(False)
      
      
    def bestK(self):
        """@brief   Runs K-Means multiple times to find the best K for the current 
                    data given the 'fitting' method. In cas of Fisher elbow method 
                    is recommended.
                    
                    at the end, self.centroids and self.clusters contains the 
                    information for the best K. NO need to rerun KMeans.
           @return B is the best K found.
        """
        t = time.time()
        if self.options['fitting'].lower() == 'fisher':
            fit = []
            alpha = 0.2
            for k in range(2, 15):
                self._init_rest(k)
                self.run()
                fit.append(self.fitting())
            differences = np.diff(fit)
            
            lastDiff = differences[0]
            for i in range(1,len(differences)):
                if differences[i] >= lastDiff*alpha:
                    self.K = i + 1
                    break
                else:
                    lastDiff = differences[i]
            
        if self.options['fitting'].lower() == 'silhouette':
            fit = []
            for k in range(2, 15):
                self._init_rest(k)
                self.run()
                fit.append(abs(1-self.fitting()))
            self.K = np.argmin(fit) + 2
            print "BestK --> " + str(self.K)

        
        print ""
        print self.options['fitting'] + " TIME --> " + str(time.time() - t)
        
    def fitting(self):
        """@brief  return a value describing how well the current kmeans fits the data
        """
        if self.K < 2:
            return np.inf
        
        #Intraclass distance calculation
        intraClassDistance = []
        im_centroids_distance = distance(self.X, self.centroids)
        for k in range(self.K):
            intraDistance = np.mean(im_centroids_distance[self.clusters==k][:,k])
            if not np.isnan(intraDistance):
                intraClassDistance.append(intraDistance)
        intraClassDistance = np.mean(intraClassDistance)

        #Interclass distance calculation
        allPixelsMean = np.mean(self.X, axis=0)
        center_centroids_distance = distance(np.array(allPixelsMean), self.centroids)
        interClassDistance = np.mean(center_centroids_distance)
                    
        if self.options['fitting'].lower() == 'fisher':
            result =  intraClassDistance/interClassDistance
        if self.options['fitting'].lower() == 'silhouette':
            result =  (interClassDistance-intraClassDistance)/max(intraClassDistance,interClassDistance)
            
        return result


    def plot(self, first_time=True):
        """@brief   Plots the results
        """

        #markersshape = 'ov^<>1234sp*hH+xDd'    
        markerscolor = 'bgrcmybgrcmybgrcmyk'
        if first_time:
            plt.gcf().add_subplot(111, projection='3d')
            plt.ion()
            plt.show()

        if self.X.shape[1]>3:
            if not hasattr(self, 'pca'):
                self.pca = PCA(n_components=3)
                self.pca.fit(self.X)
            Xt = self.pca.transform(self.X)
            Ct = self.pca.transform(self.centroids)
        else:
            Xt=self.X
            Ct=self.centroids

        for k in range(self.K):
            plt.gca().plot(Xt[self.clusters==k,0], Xt[self.clusters==k,1], Xt[self.clusters==k,2], '.'+markerscolor[k])
            plt.gca().plot(Ct[k,0:1], Ct[k,1:2], Ct[k,2:3], 'o'+'k',markersize=12)

        if first_time:
            plt.xlabel('dim 1')
            plt.ylabel('dim 2')
            plt.gca().set_zlabel('dim 3')
        plt.draw()
        plt.pause(0.01)