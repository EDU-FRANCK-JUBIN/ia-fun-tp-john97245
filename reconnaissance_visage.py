# -*- coding: utf-8 -*-
"""
Created on Thu Feb 13 14:48:29 2020

@author: johna
"""

import cv2
import sys
from matplotlib import pyplot as plt

def frontal_face_default():
        imagePath = r'images/image0.jpg'
        #fichier contenant les fichiers xml par défaut
        dirCascadeFiles = r'./opencv/haarcascades_cuda/'
        #xml de la  renconnaissance faciale
        cascadefile = dirCascadeFiles + "haarcascade_frontalface_default.xml"
        classCascade = cv2.CascadeClassifier(cascadefile)
        print (classCascade)
        image = cv2.imread(imagePath)
        ## grise l'image pour augmenter l'efficacité de la reconnaissance faciale
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        faces = classCascade.detectMultiScale(
                gray,
                scaleFactor=1.1,
                minNeighbors=5,
                minSize=(30,30),
                flags = cv2.CASCADE_SCALE_IMAGE
                )
        print("Il y a {0} visage(s).".format(len(faces)))
        
        # Dessine des rectangles autour des visages
        for(x,y,w,h) in faces:
            cv2.rectangle(image, (x,y), (x+w,y+h), (0,255,0),2)
        
        plt.imshow(image)
        

def frontal_face_alt():
        imagePath = r'images/image0.jpg'
        #fichier contenant les fichiers xml par défaut
        dirCascadeFiles = r'./opencv/haarcascades_cuda/'
        #xml de la  renconnaissance faciale
        cascadefile = dirCascadeFiles + "haarcascade_frontalface_alt.xml"
        classCascade = cv2.CascadeClassifier(cascadefile)
        print (classCascade)
        image = cv2.imread(imagePath)
        ## grise l'image pour augmenter l'efficacité de la reconnaissance faciale
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        faces = classCascade.detectMultiScale(
                gray,
                scaleFactor=1.1,
                minNeighbors=5,
                minSize=(30,30),
                flags = cv2.CASCADE_SCALE_IMAGE
                )
        print("Il y a {0} visage(s).".format(len(faces)))
        
        # Dessine des rectangles autour des visages
        for(x,y,w,h) in faces:
            cv2.rectangle(image, (x,y), (x+w,y+h), (0,255,0),2)
        plt.imshow(image)
        # récupère seulement le rectangle
        i=0
        for(x,y,w,h) in faces:
           crop_img = image[y:y+h, x:x+w]
           #sauvegarde l'image dans un fichier
           cv2.imwrite('fichier_resultat_'+str(i)+'.jpg', image[y:y+h, x:x+w])
           i = i+1
        #plt.imshow(crop_img)
        


def frontal_face_alt_multi_faces():
        imagePath = r'images/image3.jpg'
        #fichier contenant les fichiers xml par défaut
        dirCascadeFiles = r'./opencv/haarcascades_cuda/'
        #xml de la  renconnaissance faciale
        cascadefile = dirCascadeFiles + "haarcascade_frontalface_default.xml"
        classCascade = cv2.CascadeClassifier(cascadefile)
        print (classCascade)
        image = cv2.imread(imagePath)
        ## grise l'image pour augmenter l'efficacité de la reconnaissance faciale
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        faces = classCascade.detectMultiScale(
                gray,
                scaleFactor=1.1,
                minNeighbors=5,
                minSize=(30,30),
                flags = cv2.CASCADE_SCALE_IMAGE
                )
        print("Il y a {0} visage(s).".format(len(faces)))
        
        # Dessine des rectangles autour des visages
        for(x,y,w,h) in faces:
            cv2.rectangle(image, (x,y), (x+w,y+h), (0,255,0),2)
        #plt.imshow(image)
        # récupère seulement le rectangle
        i=0
        for(x,y,w,h) in faces:
           crop_img = image[y:y+h, x:x+w]
           #sauvegarde l'image dans un fichier
           cv2.imwrite('fichier_resultat_'+str(i)+'.jpg', image[y:y+h, x:x+w])
           i = i+1
           
        # récupérer les coordonnées sur l'image
        for i in range(len(faces)):
            print("Cadre du visage N°{0} --> {1}".format(i,faces[i]))
        #plt.imshow(crop_img)
        
        for i in range(len(faces)):
            plt.subplot(1,2, i+1)
            plt.imshow(image[faces[i][1]:faces[i][1]+faces[i][3], faces[i][0]:faces[i][0]+faces[i][2]])
frontal_face_alt_multi_faces()