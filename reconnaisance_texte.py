# -*- coding: utf-8 -*-
"""
Created on Thu Feb 13 16:53:40 2020

@author: johna
"""

# OCR : (Optical Character Recognition)
from pdf2image import convert_from_path, convert_from_bytes
try:
    from PIL import Image
except ImportError:
    import Image
import pytesseract as pyt
from pytesseract import Output
import cv2
from matplotlib import pyplot as plt

# cmd to set up the tesseract exe 
pyt.pytesseract.tesseract_cmd = r'C:\Users\johna\AppData\Local\Tesseract-OCR\tesseract.exe'

imageF = r'images/test_code.jpg'

##### Funtions ######

# grayscale
def grayscale(image):
    return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# noise removal
def remove_noise(image):
    return cv2.medianBlur(image,5)

# thresholding
def thresholding(image):
    return cv2.threshold(image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]

#######################
    

def img_to_str(image):
    print(pyt.image_to_string(Image.open(image)))
img_to_str(imageF)
    
def img_to_str_details(image):
    simage = imageF
    img = cv2.imread(simage)
    d = pyt.image_to_data(img,output_type=Output.DICT)
    #print(d)
    NbBoites = len(d['level'])
    print("Nombre de boites ! "+ str(NbBoites))
    for i in range(NbBoites):
        # Récupère les coordonnées de chaque boite
        (x,y,w,h) = (d['left'][i], d['top'][i],d['width'][i], d['height'][i])
        # Affiche un rectangle
        cv2.rectangle(img, (x,y), (x + w, y + h), (0,255,0),2)
        
    cv2.imshow('img',img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

#img_to_str_details()
    
def tesseract_avdvance_1(image):
    #simage = r'images/test_texte.jpg'
    simage = image
    img = cv2.imread(simage)
    
    #remove some noise
    retouche = remove_noise(img)
    
    # display the original img
    plt.imshow(retouche,'gray')
    
    print(pyt.image_to_string(retouche))
    
    
def tesseract_avdvance_2(image):
    #simage = r'images/test_texte.jpg'
    simage = image
    img = cv2.imread(simage)
    
    #remove some noise
    retouche2 = thresholding(grayscale(remove_noise(img)))
    
    # display the original img
    plt.imshow(retouche2,'gray')
    
    print(pyt.image_to_string(retouche2))
   

#Tesseract_avdvance_2(imageF)
#img_to_str(r'images/test.jpg')
    
 
def pdf_to_img(pdf):
    #simage = r'images/test_texte.jpg'
    img = convert_from_path(pdf)
    
    print("Nombre de pages "+ str(len(img)))
    for i in range(len(img)):
        print("Page N°" + str(i+1) + "\n")
        print(pyt.image_to_string(img[i]))  
        
#pdf_to_img(r'images/attestationCvec.pdf')

