#!/usr/bin/python
import matplotlib.pyplot as plt
import numpy as np
import cv2


def plot_images_labels_prediction(file_names, images, labels, preds, values, s=(4,4), pix=(64, 64)):
    fig, ax = plt.subplots(s[0], s[1])
    num = int(s[0]*s[1])
    for index in range(int(len(labels)/num) + 1):
        fig = plt.gcf() # Get Current Figure
        fig.set_size_inches(10, 12)
        n = len(labels) - index*(num)
        n = num if n > num else n
        for i in range(n):
            j = index*num + i
            ax = plt.subplot(s[0], s[1], i+1)
            im = cv2.resize(images[j], pix)
            ax.imshow(im)
            title = file_names[j] + '\n' + labels[j] + '-->' + preds[j] + '(' +'%.3f'%values[j]+')'
            ax.set_title(title,fontsize=10)
            ax.set_xticks([])
            ax.set_yticks([])        
        plt.show()

