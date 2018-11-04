# -*- coding: utf-8 -*-
"""

@author: ramon
"""
from skimage import io
from skimage.transform import rescale
import matplotlib.pyplot as plt


import Labels as lb


plt.close("all")
if __name__ == "__main__":

    #'colorspace': 'RGB', 'Lab' o 'ColorNaming'
    #'fitting': 'fisher' o 'silhouette'
    options = {'colorspace':'ColorNaming', 'K':0, 'synonyms':False, 'single_thr':0.6, 
               'verbose':False, 'km_init':'First', 'metric':'Basic', 
               'fitting':'Fisher', 'tolerance':0}

    ImageFolder = 'Images'
    GTFile = 'LABELSsmall.txt'
    
    GTFile = ImageFolder + '/' + GTFile
    GT = lb.loadGT(GTFile)

    DBcolors = []
    for gt in GT:
        print gt[0]
        im = io.imread(ImageFolder+"/"+gt[0])
        im = rescale(im, 0.2, preserve_range=True)
        colors,_,_ = lb.processImage(im, options)
        DBcolors.append(colors)
        
    basicEncert,_ = lb.evaluate(DBcolors, GT, options)
    options['metric'] = 'Complete'
    completeEncert,_ = lb.evaluate(DBcolors, GT, options)
    
    print "Encert promig: "+ '%.2f' % (basicEncert*100) + '%'
    print "Puntuació total promig: " + '%.2f' % (completeEncert*100) + '%'
