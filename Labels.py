# -*- coding: utf-8 -*-
"""
@author: Xavier Velasco Llaurado
"""
import re
import numpy as np
import ColorNaming as cn
from skimage import color
import KMeans as km


def loadGT(fileName):
    """@brief   Loads the file with groundtruth content
    
    @param  fileName  STRING    name of the file with groundtruth
    
    @return groundTruth LIST    list of tuples of ground truth data
                                (Name, [list-of-labels])
    """

    groundTruth = []
    fd = open(fileName, 'r')
    for line in fd:
        splitLine = line.split(' ')[:-1]
        labels = [''.join(sorted(filter(None,re.split('([A-Z][^A-Z]*)',l)))) for l in splitLine[1:]]
        groundTruth.append( (splitLine[0], labels) )
        
    return groundTruth


def evaluate(description, GT, options):
    """@brief   EVALUATION FUNCTION
    @param description LIST of color name lists: contain one lsit of color labels for every images tested
    @param GT LIST images to test and the real color names (see  loadGT)
    @options DICT  contains options to control metric, ...
    @return mean_score,scores mean_score FLOAT is the mean of the scores of each image
                              scores     LIST contain the similiraty between the ground truth list of color names and the obtained
    """
    scores = []

    for i in range(len(description)):
        scores.append(similarityMetric(description[i], GT[i][1], options))
        print "==== IMATGE " + GT[i][0] + " ===="
        print "COLORSPACE ---> " + options['colorspace']
        print "INIT ---------> " + options['km_init']
        print "KMEANS -------> " + str(description[i])
        print "GROUNDTRUTH --> " + str(GT[i][1])
        print "================================"

    
    return np.mean(np.array(scores)), scores


def similarityMetric(Est, GT, options):
    """@brief   SIMILARITY METRIC
    @param Est LIST  list of color names estimated from the image ['red','green',..]
    @param GT LIST list of color names from the ground truth
    @param options DICT  contains options to control metric, ...
    @return S float similarity between label LISTs
    """
    
    if options == None:
        options = {}
    if not 'metric' in options:
        options['metric'] = 'basic'
        
    matchingColors =  set(Est).intersection(GT)
    if options['metric'].lower() == 'basic'.lower():
        return float(len(matchingColors)) / float(len(Est))
    if options['metric'].lower() == 'complete'.lower():
        return float(len(matchingColors)) / float(len(GT))
    if options['metric'].lower() == 'SOF'.lower():
        return float(2.0*len(matchingColors) / (len(Est) + len(GT)))
    else:
        return 0.5
        
def getLabels(kmeans, options):
    """@brief   Labels all centroids of kmeans object to their color names
    
    @param  kmeans  KMeans      object of the class KMeans
    @param  options DICTIONARY  options necessary for labeling
    
    @return colors  LIST    colors labels of centroids of kmeans object
    @return ind     LIST    indexes of centroids with the same color label
    """

    #  remind to create composed labels if the probability of 
    #  the best color label is less than  options['single_thr']
    
    colors = []
    cluster_count = np.array([])
    ind = []
    centroid_colors = {color: [] for color in cn.colors}
    
    _, cluster_count = np.unique(kmeans.clusters, return_counts=True)
    
    for k in range(kmeans.K):
        most_likely_colors = []
        most_likely_colors_indexes = []
        
        # Cluster amb mes pixels
        index = np.argmax(cluster_count)    
        # Probabilitats dels dos colors que mes apareixen en aquest cluster
        most_likely_colors_probabilities = np.flip(np.sort(kmeans.centroids[index]), axis=0)[:2]
        # Indexos a cn.colors dels dos colors amb mes probabilitat
        most_likely_colors_indexes.append(np.where(kmeans.centroids[index]==most_likely_colors_probabilities[0])[0][0])
        most_likely_colors_indexes.append(np.where(kmeans.centroids[index]==most_likely_colors_probabilities[1])[0][0])
        # Nom dels colors amb mes probabilitat
        most_likely_colors.append(cn.colors[most_likely_colors_indexes[0]])
        most_likely_colors.append(cn.colors[most_likely_colors_indexes[1]])
        
        if most_likely_colors_probabilities[0] > options['single_thr']:
            # Assignem clusters a nom de color
            # centroid_colors['blue'] = [1, 3]
            centroid_colors[most_likely_colors[0]].append(index)
            # Si el nom de color no es a la llista de colors a retornar el fiquem
            if most_likely_colors[0] not in colors:
                colors.append(most_likely_colors[0])
        else:
            # Creem el color compost per ordre alfabetic
            if most_likely_colors[0] > most_likely_colors[1]:
                compound_color = most_likely_colors[1] + most_likely_colors[0]
            else:
                compound_color = most_likely_colors[0] + most_likely_colors[1]
            # Si el color compost no es al diccionari de centroids (de primeres nomes hi ha colors normals) 
            # n'afegim una llista amb numero de cluster dins
            # Si ja hi es, hi fiquem el numero de cluster sense llista (la llista ja esta creada)
            if compound_color not in centroid_colors:
                centroid_colors[compound_color]= [index]
            else:
                centroid_colors[compound_color].append(index)
            # Si el nom del color compost no es a la llist de colors a retornar el fiquem
            if compound_color not in colors:
                colors.append(compound_color)
        # Posem el comptador del cluster en questio a 0 per a que n'agafi el seguent mes gran
        cluster_count[index] = 0
        
    # Per cada color dins la llsita de colors a retornar n'agafem la llista de clusters que li correspon
    for c in colors:
        ind.append(centroid_colors[c])
        
    return colors, ind


def processImage(im, options):
    """@brief   Finds the colors present on the input image
    
    @param  im      LIST    input image
    @param  options DICTIONARY  dictionary with options
    
    @return colors  LIST    colors of centroids of kmeans object
    @return indexes LIST    indexes of centroids with the same label
    @return kmeans  KMeans  object of the class KMeans
    """

    #  1- CHANGE THE IMAGE TO THE CORRESPONDING COLOR SPACE FOR KMEANS
    if options['colorspace'].lower() == 'ColorNaming'.lower():  
        im = cn.ImColorNamingTSELabDescriptor(im)
    elif options['colorspace'].lower() == 'HSV'.lower():        
        im = color.rgb2hsv(im)
    elif options['colorspace'].lower() == 'Lab'.lower():        
        im = color.rgb2lab(im)

    #  2- APPLY KMEANS ACCORDING TO 'OPTIONS' PARAMETER
    kmeans = km.KMeans(im, options['K'], options) 
    kmeans.run()

    #  3- GET THE NAME LABELS DETECTED ON THE 11 DIMENSIONAL SPACE
    if options['colorspace'].lower() == 'RGB'.lower():
        kmeans.centroids = cn.ImColorNamingTSELabDescriptor(kmeans.centroids)
    elif options['colorspace'].lower() == 'HSV'.lower():
        kmeans.centroids = kmeans.centroids.reshape((1, -1, 3))
        kmeans.centroids = color.hsv2rgb(kmeans.centroids)*255.0
        kmeans.centroids = kmeans.centroids.reshape(-1, 3)
        kmeans.centroids = cn.ImColorNamingTSELabDescriptor(kmeans.centroids)
        kmeans.centroids = np.reshape(kmeans.centroids, (-1, kmeans.centroids.shape[-1]))
    elif options['colorspace'].lower() == 'Lab'.lower():
        kmeans.centroids = kmeans.centroids.reshape((1, -1, 3))
        kmeans.centroids = color.lab2rgb(kmeans.centroids)*255.0
        kmeans.centroids = kmeans.centroids.reshape(-1, 3)
        kmeans.centroids = cn.ImColorNamingTSELabDescriptor(kmeans.centroids)
        kmeans.centroids = np.reshape(kmeans.centroids, (-1, kmeans.centroids.shape[-1]))


    colors, which = getLabels(kmeans, options)   
    return colors, which, kmeans

def NIUs():
    return 1423495, 1426640, 1459038