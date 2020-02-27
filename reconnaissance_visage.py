# -*- coding: utf-8 -*-
"""
Created on Thu Feb 13 14:48:29 2020

@author: johna
"""
from pdf2image import convert_from_path, convert_from_bytes
try:
    from PIL import Image
except ImportError:
    import Image
import pytesseract as pyt
from pytesseract import Output
import cv2
from matplotlib import pyplot as plt
import sys
import numpy as np

# Fonctions de tesseract #

# cmd to set up the tesseract exe 
pyt.pytesseract.tesseract_cmd = r'C:\Users\johna\AppData\Local\Tesseract-OCR\tesseract.exe'


def img_to_str(image):
    print(pyt.image_to_string(Image.open(image)))
#########################
    
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
        

#frontal_face_alt()

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
#frontal_face_alt_multi_faces()

def CNI():
    imagePath = r'images/CNI.jpg'
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
        cv2.rectangle(image, (x,y), (x+w,y+h), (0,255,0),4)
    plt.imshow(image)
    
    # Récupère le texte de l'image
    img_to_str(imagePath)
    
CNI()

def CNI_Target():
    imagePath = r'images/CNI.jpg'
    #fichier contenant les fichiers xml par défaut
    dirCascadeFiles = r'./opencv/haarcascades_cuda/'
    #xml de la  renconnaissance faciale
    cascadefile = dirCascadeFiles + "haarcascade_frontalface_alt.xml"
    classCascade = cv2.CascadeClassifier(cascadefile)
    print (classCascade)
    # Approche ciblé (récupération d'une partie de l'image)
    image = cv2.imread(imagePath)
    x = 600
    y = 130
    w = 420
    h = 200
    #plt.imshow(cv2.rectangle(image, (x,y), (w,h), (0,255,0), 3))
    region_nom = image[y:y+h-50, x:x+w]
    plt.imshow(region_nom)
    
    # Récupère le texte de l'image
    prenomCI = pyt.image_to_string(region_nom)
    print(prenomCI)
    
#CNI_Target()