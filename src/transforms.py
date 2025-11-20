import torchvision  
import torch
import cv2 as cv
import numpy as np

#[@Qntmth-Uv] Here are the functions to create the Low and High 
# frecuency imagenes.

#Kernel size of the original paper
#src = np.zeros((256, 320, 3))
#kernel_size = 15

#Transform using for the gussian.
#x = torchvision.transforms.GaussianBlur(kernel_size, sigma=3) #Torchvision
#src = cv.GaussianBlur(src, (3, 3), 0) #OpenCV

#Transform using the Lagrangean
#src_gray = cv.cvtColor(src, cv.COLOR_BGR2GRAY)
#dst = cv.Laplacian(src_gray, cv.CV_8U, ksize=3) #OpenCV

def _get_LF_Image(Image:torch.Tensor)->torch.Tensor:
    """Function to get the low frecuency of the image using a gaussian blur
    transfor with a kernel size of 15x15 and a standar desviation of 3"""
    image = Image.cpu().numpy()
    low_frecuency = cv.GaussianBlur(image, (15, 15), 0) #OpenCV
    low_frecuency = torch.Tensor(low_frecuency).to(Image.device)
    return low_frecuency

def _get_HF_Image(Image:torch.Tensor)->torch.Tensor:
    """Function to get the low frecuency of the image using a Laplace 
    transform with a kernel size of 3x3 (supposses that the image s in a format
    of integer without sing)"""
    image = Image.cpu().numpy()
    high_frecuency_gray = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
    high_frecuency = cv.Laplacian(high_frecuency_gray, cv.CV_8U, ksize=3) #OpenCV
    high_frecuency = torch.Tensor(high_frecuency).to(Image.device)
    return high_frecuency
