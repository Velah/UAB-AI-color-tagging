# -*- coding: utf-8 -*-
"""
Created on Tue May 22 18:24:54 2018

@author: Xavier Velasco Llaurado
"""

from skimage import io
import time
from skimage.transform import rescale
import matplotlib.pyplot as plt
import numpy as np
import os
import Labels as lb

resultsFolder = 'Results'

barColors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#bcbd22', '#d62728', 'grey', 'purple', 'pink']
imageRescale = None
maxK = 15
    
fittings = ['Fisher', 'Silhouette']
colorspaces = ['RGB', 'Lab', 'ColorNaming', 'HSV']
inits = ['First', 'Random', 'Diagonal']
metrics = ['Basic', 'Complete', 'SOF']

plt.ioff()

def resultats():
    ImageFolder = 'Images'
    file = 'LABELSlarge.txt'
    GTFile = ImageFolder + '/' + file
    GT = lb.loadGT(GTFile)
    
    results = {}
    resultsK = {}
    
    for fitting in fittings:
        results[fitting] = {}
        for colorspace in colorspaces:
            results[fitting][colorspace] = {}
            for init in inits:
                results[fitting][colorspace][init] = {}
                options = {'colorspace':colorspace, 'K':0, 'km_init':init, 
                           'verbose':False, 'single_thr':0.6, 'fitting':fitting, 'max_iter': 100}
                DBcolors = []
                for gt in GT:
                    im = io.imread(ImageFolder+"/"+gt[0])
                    if imageRescale:
                        im = rescale(im, imageRescale, preserve_range=True)
                    print ""
                    print "Processing --> " + colorspace + " + " + init + " + " + fitting + " --> Image " + gt[0]  
                    print ""
                    colors,_,_ = lb.processImage(im, options)
                    DBcolors.append(colors)
                for metric in metrics:
                    options['metric'] = metric
                    encert, _ = lb.evaluate(DBcolors, GT, options)
                    results[fitting][colorspace][init][metric] = encert*100
    

    for colorspace in colorspaces:
        resultsK[colorspace] = {}
        for init in inits:
            resultsK[colorspace][init] = {}
            for metric in metrics:
                resultsK[colorspace][init][metric] = {}
            for k in range(2, maxK+1):
                options = {'colorspace':colorspace, 'K':k, 'km_init':init, 
                           'verbose':False, 'single_thr':0.6, 'max_iter': 100}
                DBcolors = []
                for gt in GT:
                    im = io.imread(ImageFolder+"/"+gt[0])
                    if imageRescale:
                        im = rescale(im, imageRescale, preserve_range=True)
                    colors,_,_ = lb.processImage(im, options)
                    DBcolors.append(colors)
                for metric in metrics:
                    options['metric'] = metric
                    encert, _ = lb.evaluate(DBcolors, GT, options)
                    resultsK[colorspace][init][metric][k] = encert*100
                    
    return results, resultsK

def plot(results, resultsK):
    if not os.path.exists(resultsFolder):
        os.makedirs(resultsFolder)
        
    # ---- FITTING PLOTTING ---- #    
    
    fittingValues = {fitting: [] for fitting in fittings}
    colorspaceValues = {colorspace: [] for colorspace in colorspaces}
    initValues = {init: [] for init in inits}
    
    for fitting in fittings:
        for colorspace in colorspaces:
            for init in inits:
                plt.figure(figsize=(8,5), frameon=False)
                ax = plt.gca()
                for x,metric in enumerate(metrics):
                    value = results[fitting][colorspace][init][metric]
                    if metric == 'Complete':
                        fittingValues[fitting].append(value)
                        colorspaceValues[colorspace].append(value)
                        initValues[init].append(value)
                    plt.bar(x, value, 0.8)
                    ax.text(x - 0.15, value + 3, str(np.around(value, decimals=2)) + "%", fontweight='bold', fontsize=14)
                plt.ylabel('% SIMILARITY')
                plt.xlabel('Metric')
                plt.title(fitting + "|" + colorspace + "|" + init)
                plt.xticks(range(len(metrics)), metrics)
                ax.set_ylim([0,100])
                plt.savefig('Results/'+fitting+"_"+colorspace+"_"+init+'.png')
                
    for key,value in fittingValues.items():
        fittingValues[key] = np.mean(value)
    for key,value in colorspaceValues.items():
        colorspaceValues[key] = np.mean(value)
    for key,value in initValues.items():
        initValues[key] = np.mean(value)   
    
    plt.figure(figsize=(8,5), frameon=False)
    ax = plt.gca()
    ax.set_ylim([0,100])
    for x,y in enumerate(fittingValues.values()):
        ax.text(x - 0.1, y + 3, str(np.around(y, decimals=2)) + "%", fontweight='bold', fontsize=14)
    plt.bar(range(len(fittingValues)), fittingValues.values(), 0.8, color=barColors)
    plt.xticks(range(len(fittingValues)), fittingValues.keys())
    plt.ylabel('% SIMILARITY --> Complete metric')
    plt.title('Fittings')
    plt.savefig('Results/Fittings.png')
    
    plt.figure(figsize=(8,5), frameon=False)
    ax = plt.gca()
    ax.set_ylim([0,100])
    for x,y in enumerate(colorspaceValues.values()):
        ax.text(x - 0.2, y + 3, str(np.around(y, decimals=2)) + "%", fontweight='bold', fontsize=14)
    plt.bar(range(len(colorspaceValues)), colorspaceValues.values(), 0.8, color=barColors)
    plt.xticks(range(len(colorspaceValues)), colorspaceValues.keys())
    plt.ylabel('% SIMILARITY --> Complete metric')
    plt.title('Color Spaces')
    plt.savefig('Results/ColorSpaces.png')
    
    plt.figure(figsize=(8,5), frameon=False)
    ax = plt.gca()
    ax.set_ylim([0,100])
    for x,y in enumerate(initValues.values()):
        ax.text(x - 0.15, y + 3, str(np.around(y, decimals=2)) + "%", fontweight='bold', fontsize=14)
    plt.bar(range(len(initValues)), initValues.values(), 0.8, color=barColors)
    plt.xticks(range(len(initValues)), initValues.keys())
    plt.ylabel('% SIMILARITY --> Complete metric')
    plt.title('Inits')
    plt.savefig('Results/Inits.png')
    
    
    # ---- K PLOTTING ---- #
    
    colorspaceValuesK = {colorspace: {} for colorspace in colorspaces}
    for key in colorspaceValuesK.keys():
        for k in range(2, maxK+1):
            colorspaceValuesK[key][k] = []
    initValuesK = {init: {} for init in inits}
    for key in initValuesK.keys():
        for k in range(2, maxK+1):
            initValuesK[key][k] = []
    metricValuesK = {metric: {} for metric in metrics}
    for key in metricValuesK.keys():
        for k in range(2, maxK+1):
            metricValuesK[key][k] = []
    
    for colorspace in colorspaces:
        for init in inits:
            for metric in metrics:
                fig = plt.figure(figsize=(8,5), frameon=False)
                ax = plt.gca()
                for k in range(2, maxK+1):
                    x = k-2
                    value = resultsK[colorspace][init][metric][k]
                    metricValuesK[metric][k].append(value)
                    if metric == 'Complete':
                        colorspaceValuesK[colorspace][k].append(value)
                        initValuesK[init][k].append(value)
                    plt.bar(x, value, 0.8)
                plt.ylabel('% SIMILARITY')
                plt.xlabel('K')
                plt.title(colorspace + "|" + init + "|" + metric)
                plt.xticks(range(maxK), range(2,maxK+1))
                ax.set_ylim([0,100])
                plt.savefig('Results/'+colorspace+"_"+init+"_"+metric+"_"+'K'+'.png')
                plt.close(fig)
                
    for key in colorspaceValuesK.keys():
        for k,value in colorspaceValuesK[key].items():
            colorspaceValuesK[key][k] = np.mean(value)
    for key in initValuesK.keys():
        for k,value in initValuesK[key].items():
            initValuesK[key][k] = np.mean(value)   
    for key in metricValuesK.keys():
        for k,value in metricValuesK[key].items():
            metricValuesK[key][k] = np.mean(value)
    
    fig = plt.figure(figsize=(8,5), frameon=False)
    ax = plt.gca()
    ax.set_ylim([0,100])
    for key in colorspaceValuesK.keys():
        plt.plot(colorspaceValuesK[key].values(), label=key)
    plt.legend(loc='upper right')
    plt.xticks(range(maxK), range(2,maxK+1))
    plt.ylabel('% SIMILARITY --> Complete metric')
    plt.xlabel('K')
    plt.title('Color Spaces')
    plt.savefig('Results/ColorSpacesK.png')
    plt.close(fig)
    
    fig = plt.figure(figsize=(8,5), frameon=False)
    ax = plt.gca()
    ax.set_ylim([0,100])
    for key in initValuesK.keys():
        plt.plot(initValuesK[key].values(), label=key)
    plt.legend(loc='upper right')
    plt.xticks(range(maxK), range(2,maxK+1))
    plt.ylabel('% SIMILARITY --> Complete metric')
    plt.xlabel('K')
    plt.title('Inits')
    plt.savefig('Results/InitsK.png')
    plt.close(fig)
    
    fig = plt.figure(figsize=(8,5), frameon=False)
    ax = plt.gca()
    ax.set_ylim([0,100])
    for key in metricValuesK.keys():
        plt.plot(metricValuesK[key].values(), label=key)
    plt.legend(loc='upper right')
    plt.xticks(range(maxK), range(2,maxK+1))
    plt.ylabel('% SIMILARITY')
    plt.xlabel('K')
    plt.title('Metrics')
    plt.savefig('Results/MetricsK.png')
    plt.close(fig)

def main():
    plt.close("all")
    t=time.time()
    results, resultsK = resultats()
    plot(results, resultsK)
    plt.close("all")
    print ""
    print "Els resultats s'han guardat a la carpeta " + resultsFolder
    print "TEMPS --> " + str(time.time() - t)
    
    
if __name__ == '__main__':
    main()  
    
                        