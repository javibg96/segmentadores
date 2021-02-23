from skimage import segmentation, color, future, io, util, morphology, measure
from skimage.segmentation import slic, felzenszwalb, watershed, random_walker, relabel_sequential
from skimage.exposure import histogram
import matplotlib.pyplot as plt
import numpy as np
from skimage.util import img_as_float
from skimage.filters import rank, gaussian, threshold_otsu
from scipy import ndimage as ndi
from skimage.morphology import disk, square
from skimage.future import graph
import cv2
import os
from statistics import mode


from skimage.metrics import (adapted_rand_error,
                              variation_of_information)


folder="PATH"

folder_g_t = "PATH"



file1= open("votation.txt", "a")

def factor_f(image, ground_truth):
    if (image.shape != ground_truth.shape):
        ground_truth =  cv2.cvtColor(ground_truth, cv2.COLOR_BGR2GRAY)
    assert image.shape == ground_truth.shape
    relab_gt, fw_gt, inv_gt = relabel_sequential(ground_truth)
    relab, fw, inv = relabel_sequential(image)
    error, precision, recall = adapted_rand_error(relab_gt, relab)
    factor_f = 2*(precision*recall)/(precision+recall)
    return factor_f



def load_images_from_folder(folder):
    images = []
    list_files = os.listdir(folder)
    list_files.sort()
    for filename in list_files:
        print(filename)
        img = cv2.imread(os.path.join(folder,filename))
        if img is not None:
            images.append(img)
    return images

def apply_k_means(img):  
	seg = slic(gaussian(img,5), n_segments= 8 , compactness=10 )
	res = segmentation.mark_boundaries(img, seg)
	return seg

def grafo(img):
  labels = segmentation.slic(gaussian(img,5), n_segments= 20, compactness=10, multichannel=True)
  rag = graph.rag_mean_color(img, labels, mode='similarity')
  new_labels = graph.cut_normalized(labels, rag)
  rec = segmentation.mark_boundaries(img, new_labels)
 # new_labels = color.label2rgb(new_labels)
  return new_labels

def active_contours(img):
  gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
  gray = cv2.medianBlur(gray, 9)  # 90%
  #gray = cv2.bilateralFilter(gray,9,125,125)
  flag, thresh = cv2.threshold(gray,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)


  # Find contours
  contours, hierarchy = cv2.findContours(thresh,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
  contours = sorted(contours, key=cv2.contourArea,reverse=True) 
  # Select long perimeters only
  perimeters = [cv2.arcLength(contours[i],True) for i in range(len(contours))]
  listindex=[i for i in range(15) if perimeters[i]>perimeters[0]/2]
  numcards=len(listindex)
  # Show image
  imgcont = img.copy()
  [cv2.drawContours(imgcont, [contours[i]], 0, (0,255,0), 5) for i in listindex]
  return imgcont

def apply_watershed(img):
    
    # Binarizamos la imagen con la binarizacion OTSU
    bw = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    _, bw = cv2.threshold(bw, 40, 255, cv2.THRESH_OTSU)
    
    # Buscamos la distancia euclidia y la normalizamos
    dist = cv2.distanceTransform(bw, cv2.DIST_L2, 3)
    dist = cv2.normalize(dist, dist, 0, 1.0, cv2.NORM_MINMAX)
    euclidian_dist = dist
    
    # Binarizamos la distancia euclidia
    _, dist = cv2.threshold(dist, 0.1, 1.0, cv2.THRESH_BINARY)
    # Realizamos una dilatacion
    kernel1 = np.ones((3,3), dtype=np.uint8)
    dist = cv2.dilate(dist, kernel1)
    
    # Encontramos los contornos 
    dist_8u = dist.astype('uint8')
    contours, _= cv2.findContours(dist_8u, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    
    markers = np.zeros(dist.shape, dtype=np.int32)
    # Dibujamos los contornos
    for i in range(len(contours)):
        cv2.drawContours(markers, contours, i, (i+1), -1)
    
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Aplicamos el watershed
    segmentations = watershed(img, markers)
    #segmentations = color.label2rgb(segmentations)
    
    return segmentations

def apply_random_walker(img):
    img_f = img_as_float(img)
    markers = np.zeros(img_f.shape, dtype=np.uint)
    markers[img_f < 0.6 ] = 1
    markers[img_f >= 0.7] = 2
    labels = random_walker(img, markers, beta=10, mode='bf')
    labels[labels ==1] = 0
    labels[labels == 2] = 255
    return labels

def apply_felzenwalb(img):
    segments = felzenszwalb(gaussian(img,10), scale=300, sigma=0.1, min_size=300)
   # segments =  color.label2rgb(segments)
    return segments


def morf_seg(img):
  img = color.rgb2gray(img)
  Thresh = threshold_otsu(img)
  img = img > Thresh

  Strel = morphology.disk(5)
  img = morphology.opening(img,Strel)
  img = morphology.closing(img,Strel)

  img = img_as_ubyte(img)

  res = cv2.findContours(img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
  contours = res[-2]

  cv2.fillPoly(img,pts=contours,color=(255,255,255))

  img[img == 0] = 0
  img[img == 255] = 1
  return img


def run_grabcut(img_orig):
    rect_final = (100, 200, 600, 600)
    # Inicializamos la mascara
    mask = np.zeros(img_orig.shape[:2],np.uint8) 
 
    # Extract the rectangle and set the region of 
    # interest in the above mask 
    # Extraemos el rectángulo y marcamos la region de interes en la mascara 
    x,y,w,h = rect_final 
    mask[y:y+h, x:x+w] = 1 
 
    # Initialize background and foreground models 
    bgdModel = np.zeros((1,65), np.float64) 
    fgdModel = np.zeros((1,65), np.float64) 
 
    # Run Grabcut algorithm 
    cv2.grabCut(img_orig, mask, rect_final, bgdModel, fgdModel, 5, cv2.GC_INIT_WITH_RECT) 
 
    # Extract new mask 
    mask2 = np.where((mask==2)|(mask==0),0,1).astype('uint8') 
 
    # Apply the above mask to the image 
    img_orig = img_orig*mask2[:,:,np.newaxis] 
 
    # Display the image 
    return img_orig



images = load_images_from_folder(folder)
images_g_t= load_images_from_folder(folder_g_t)


def threshhold(x, thresh):
  res = np.int64(0)
  if(x>thresh):
    res = np.int64(x)
  return res

def apply_moda(datos):
    repeticiones = 0

    for i in datos:
        n = datos.count(i)
        if n > repeticiones:
            repeticiones = n

    moda = [] #Arreglo donde se guardara el o los valores de mayor frecuencia 

    for i in datos:
        n = datos.count(i) # Devuelve el número de veces que x aparece enla lista.
        if n == repeticiones and i not in moda:
            moda.append(i)

    return moda

def votation_sistem(seg_1, seg_2, seg_3, seg_4, seg_5):
    assert seg_1.shape == seg_2.shape == seg_3.shape == seg_4.shape == seg_5.shape
    result = np.zeros(seg_1.shape, dtype=np.int32)
    
    seg_1, fw, inv = relabel_sequential(seg_1)
    print(np.unique(seg_1))
    seg_2, fw, inv = relabel_sequential(seg_2)
    print(np.unique(seg_2))
    seg_3, fw, inv = relabel_sequential(seg_3)
    print(np.unique(seg_3))
    seg_4, fw, inv = relabel_sequential(seg_4)
    print(np.unique(seg_4))
    seg_5, fw, inv = relabel_sequential(seg_5)
    print(np.unique(seg_5))
    for y in range(seg_1.shape[0]):
        for x in range(seg_2.shape[1]):
            values = [seg_1[y,x], seg_2[y,x], seg_3[y,x], seg_4[y,x], seg_5[y,x]]
            #moda = threshhold(mode(values), 2)
            moda = mode(values)
            #print(type(moda))
            result[y,x] = moda
    result = color.label2rgb(result)
    return result



for i in range(len(images)):
    
     seg_w = apply_watershed(images[i])
     seg_grafos = grafo(images[i])
     seg_kmeans = apply_k_means(images[i])
     seg_fw = apply_felzenwalb(images[i])
     seg_rw= apply_random_walker(images[i])
     seg_m = morf_seg(images[i])
     #io.imsave('seg_morf'+str(i+1)+'.png', img_as_ubyte(seg_m))
     
     super_s = votation_sistem(seg_w, seg_grafos, seg_fw, seg_kmeans, seg_m)
    
     io.imsave('seg_votation'+str(i+1)+'.png', super_s)
     # = factor_f(img_as_ubyte(super_s), images_g_t[i] )
     #file1.write(str(f)+ str(" ") + "\n")


file1.close()

    
    

