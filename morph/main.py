import numpy as np
import os
import sys
import argparse
import random
from time import time
from tqdm import tqdm
from math import sqrt
import dlib
import cv2
from skimage import io
from skimage.color import rgb2hsv, hsv2rgb, rgb2yuv, yuv2rgb
from skimage.exposure import equalize_hist, equalize_adapthist
from skimage.measure import find_contours
from matplotlib import pyplot as plt
from glob import glob

from keras.models import Sequential, Model
from keras.layers import *
from keras.layers.advanced_activations import LeakyReLU
from keras.activations import relu
from keras.initializers import RandomNormal
from keras.applications import *
import keras.backend as K
import tensorflow as tf

from FCN8s_keras import FCN


def applyAffineTransform(src, srcTri, dstTri, size, inter=cv2.INTER_CUBIC) :
    # Given a pair of triangles, find the affine transform.
    warpMat = cv2.getAffineTransform( np.float32(srcTri), np.float32(dstTri) )
    # Apply the Affine Transform just found to the src image
    dst = cv2.warpAffine( src, warpMat, (size[0], size[1]), None, flags=inter, borderMode=cv2.BORDER_REFLECT_101 )
    return dst

# Warps and alpha blends triangular regions from img1 to img
def morphTriangle(img1, img, t1, t, inter=cv2.INTER_CUBIC) :
    # Find bounding rectangle for each triangle
    r1 = cv2.boundingRect(np.float32([t1]))
    r = cv2.boundingRect(np.float32([t]))
    # Offset points by left top corner of the respective rectangles
    t1Rect = []
    tRect = []
    for i in range(0, 3):
        tRect.append(((t[i][0] - r[0]),(t[i][1] - r[1])))
        t1Rect.append(((t1[i][0] - r1[0]),(t1[i][1] - r1[1])))
    # Get mask by filling triangle
    mask = np.zeros((r[3], r[2], 3), dtype = np.float32)
    cv2.fillConvexPoly(mask, np.int32(tRect), (1.0, 1.0, 1.0), 16, 0);
    # Apply warpImage to small rectangular patches
    img1Rect = img1[r1[1]:r1[1] + r1[3], r1[0]:r1[0] + r1[2]]
    size = (r[2], r[3])
    warpImage1 = applyAffineTransform(img1Rect, t1Rect, tRect, size, inter)
    # Copy triangular region of the rectangular patch to the output image
    img[r[1]:r[1]+r[3], r[0]:r[0]+r[2]] = img[r[1]:r[1]+r[3], r[0]:r[0]+r[2]] * ( 1 - mask ) + warpImage1 * mask

def pts_ind(pts, pt):
    try:
        return pts.index(pt)
    except ValueError:
        return -1

def vgg_preprocess(im):
    im = cv2.resize(im, (500, 500))
    in_ = np.array(im, dtype=np.float32)
    in_ = in_[:,:,::-1]
    in_ -= np.array((104.00698793,116.66876762,122.67891434))
    in_ = in_[np.newaxis,:]
    #in_ = in_.transpose((2,0,1))
    return in_
  
def auto_downscaling(im):
    w = im.shape[1]
    h = im.shape[0]
    while w*h >= 700*700:
        im = cv2.resize(im, (0,0), fx=0.5, fy=0.5)
        w = im.shape[1]
        h = im.shape[0]
    return im

HAAR_CASCADE_FILEPATH = 'haarcascade_frontalface_default.xml'
cascade = cv2.CascadeClassifier(HAAR_CASCADE_FILEPATH)

def get_face_bbox(image):
    image = np.asarray(image)
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    faces = cascade.detectMultiScale(gray, 1.3, 5)
    if not len(faces):
        return None
    x, y, w, h = faces[0]
    return x - 30, y - 50, x + w + 30, y + h + 80

def get_face_image(image):
    bbox = get_face_bbox(image)
    image = image.crop(bbox)
    image = image.resize((128, 128), Image.BICUBIC)
    image = image.convert("RGB")
    return image, bbox

def faceHW(mask):
    pts = np.argwhere(mask)
    return pts[:,0].max() - pts[:,0].min(), pts[:,1].max() - pts[:,1].min()

lips_left_idx = (48,49,50,51,57,58,59,60,61,62,66,67)
lips_right_idx = (51,52,53,54,55,56,57,62,63,64,65,66)

def get_normals(pts):
    norms = [None for _ in pts]
    # eyes
    for i in range(36, 48):
        left = i-1
        right = i+1
        if i == 36:
            left = 41
        if i == 41:
            right = 36
        if i == 42:
            left = 47
        if i == 47:
            right = 42
        norm_y1 = pts[i][1] - pts[left][1]
        norm_x1 = -(pts[i][0] - pts[left][0])
        len1 = sqrt(norm_y1*norm_y1 + norm_x1*norm_x1)
        norm_y1 /= len1
        norm_x1 /= len1
        norm_y2 = pts[right][1] - pts[i][1]
        norm_x2 = -(pts[right][0] - pts[i][0])
        len2 = sqrt(norm_y2*norm_y2 + norm_x2*norm_x2)
        norm_y2 /= len2
        norm_x2 /= len2
        norm_x = norm_x1 + norm_x2
        norm_y = norm_y1 + norm_y2
        len3 = sqrt(norm_y*norm_y + norm_x*norm_x)
        norm_y /= len3
        norm_x /= len3
        norms[i] = (norm_y, norm_x)
    # scull
    for i in range(17):
        left = i-1
        right = i+1
        if i == 0:
            left = 0
        if i == 16:
            right = 16
        norm_y1 = pts[i][1] - pts[left][1]
        norm_x1 = -(pts[i][0] - pts[left][0])
        len1 = sqrt(norm_y1*norm_y1 + norm_x1*norm_x1)
        if len1 != 0:
            norm_y1 /= len1
            norm_x1 /= len1
        norm_y2 = pts[right][1] - pts[i][1]
        norm_x2 = -(pts[right][0] - pts[i][0])
        len2 = sqrt(norm_y2*norm_y2 + norm_x2*norm_x2)
        if len2 != 0:
            norm_y2 /= len2
            norm_x2 /= len2
        norm_x = norm_x1 + norm_x2
        norm_y = norm_y1 + norm_y2
        len3 = sqrt(norm_y*norm_y + norm_x*norm_x)
        norm_y /= len3
        norm_x /= len3
        norms[i] = (-norm_y, -norm_x) #idk why it got switched anyway
    # lips
    for i in range(48, 68):
        left = i-1
        right = i+1
        if i == 48:
            left = 59
        if i == 59:
            right = 48
        if i == 60:
            left = 67
        if i == 67:
            right = 60
        norm_y1 = pts[i][1] - pts[left][1]
        norm_x1 = -(pts[i][0] - pts[left][0])
        len1 = sqrt(norm_y1*norm_y1 + norm_x1*norm_x1)
        norm_y1 /= len1
        norm_x1 /= len1
        norm_y2 = pts[right][1] - pts[i][1]
        norm_x2 = -(pts[right][0] - pts[i][0])
        len2 = sqrt(norm_y2*norm_y2 + norm_x2*norm_x2)
        norm_y2 /= len2
        norm_x2 /= len2
        norm_x = norm_x1 + norm_x2
        norm_y = norm_y1 + norm_y2
        len3 = sqrt(norm_y*norm_y + norm_x*norm_x)
        if len3 == 0: #3 collinear points in the lip corner
            norm_y = -norm_x1
            norm_x = norm_y1
        else:
            norm_y /= len3
            norm_x /= len3
        norms[i] = (norm_y, norm_x) #idk why they got switched
    #eyebrows
    for i in range(17, 27):
        left = i-1
        right = i+1
        if i == 17:
            left = 17
        if i == 21:
            right = 21
        if i == 22:
            left = 22
        if i == 26:
            right = 26
        norm_y1 = pts[i][1] - pts[left][1]
        norm_x1 = -(pts[i][0] - pts[left][0])
        len1 = sqrt(norm_y1*norm_y1 + norm_x1*norm_x1)
        if len1 != 0:
            norm_y1 /= len1
            norm_x1 /= len1
        norm_y2 = pts[right][1] - pts[i][1]
        norm_x2 = -(pts[right][0] - pts[i][0])
        len2 = sqrt(norm_y2*norm_y2 + norm_x2*norm_x2)
        if len2 != 0:
            norm_y2 /= len2
            norm_x2 /= len2
        norm_x = norm_x1 + norm_x2
        norm_y = norm_y1 + norm_y2
        len3 = sqrt(norm_y*norm_y + norm_x*norm_x)
        norm_y /= len3
        norm_x /= len3
        norms[i] = (norm_y, norm_x) #idk why it got switched anyway
    #nose (not actually normals)
    #27, 30, 31, 33, 35
    norm_x = pts[30][0] - pts[27][0]
    norm_y = pts[30][1] - pts[27][1]
    len30 = sqrt(norm_y*norm_y + norm_x*norm_x)
    norms[30] = (norm_x / len30, norm_y / len30)
    norms[27] = (-norm_x / len30, -norm_y / len30)

    norm_x = pts[31][0] - pts[35][0]
    norm_y = pts[31][1] - pts[35][1]
    len30 = sqrt(norm_y*norm_y + norm_x*norm_x)
    norms[31] = (norm_x / len30, norm_y / len30)
    norms[35] = (-norm_x / len30, -norm_y / len30)

    norm_y1 = pts[33][1] - pts[32][1]
    norm_x1 = -(pts[33][0] - pts[32][0])
    len1 = sqrt(norm_y1*norm_y1 + norm_x1*norm_x1)
    norm_y1 /= len1
    norm_x1 /= len1
    norm_y2 = pts[34][1] - pts[33][1]
    norm_x2 = -(pts[34][0] - pts[33][0])
    len2 = sqrt(norm_y2*norm_y2 + norm_x2*norm_x2)
    norm_y2 /= len2
    norm_x2 /= len2
    norm_x = norm_x1 + norm_x2
    norm_y = norm_y1 + norm_y2
    len3 = sqrt(norm_y*norm_y + norm_x*norm_x)
    norm_y /= len3
    norm_x /= len3
    norms[33] = (-norm_y, -norm_x)
    
    #4head
    for i in range(68, 75):
        norm_x = pts[16][0] - pts[0][0]
        norm_y = pts[16][1] - pts[0][1]
        len1 = sqrt(norm_x*norm_x + norm_y*norm_y)
        norm_x /= len1
        norm_y /= len1
        norms[i] = (-norm_y, -norm_x)
    #finally return
    return norms

def img_sparse_triang(pts_src, size):
    rect = (0, 0, size[1], size[0])
    subdiv = cv2.Subdiv2D(rect)
    pts = []
    for pt in pts_src:
        pts.append(tuple(pt))
    pts.append((0, 0))
    pts.append((0, size[0] // 2))
    pts.append((0, size[0] - 1))
    pts.append((size[1] // 2, size[0] - 1))
    pts.append((size[1] - 1, size[0] - 1))
    pts.append((size[1] - 1, size[0] // 2))
    pts.append((size[1] - 1, 0))
    pts.append((size[1] // 2, 0))
    # Insert points into subdiv
    for i in (1,4,7,9,12,15):
        subdiv.insert(pts[i])
    for i, p in enumerate(pts[17:]):
        if i+17 not in (29, 32, 34, 60, 64, 68, 74):
            subdiv.insert(p)
    triangles = subdiv.getTriangleList()
    return triangles, pts


class Morpher:
    def __init__(self, norms, pts):
        self.norms = norms
        self.pts = pts

    def eyes_size(self, img, triangles, k):
        new_pts = self.pts.copy()
        k = k / 10.
        img1 = np.float32(img.copy())
        imgMorphNorm = np.zeros(img1.shape, dtype = img1.dtype)
        new_triangles = []
        for t1 in triangles:
            t = []
            for pt in t1:
                idx = pts_ind(self.pts, pt)
                if idx in range(42, 48):
                    dist = 0.2 * sqrt((self.pts[44][0] - self.pts[24][0])**2 + (self.pts[44][1] - self.pts[24][1])**2)
                    new_x = pt[0] + int(k * dist * self.norms[idx][0])
                    new_y = pt[1] + int(k * dist * self.norms[idx][1])
                    t.append((new_x, new_y))
                    new_pts[idx] = (new_x, new_y)
                elif idx in range(36, 42):
                    dist = 0.2 * sqrt((self.pts[37][0] - self.pts[19][0])**2 + (self.pts[37][1] - self.pts[19][1])**2)
                    new_x = pt[0] + int(k * dist * self.norms[idx][0])
                    new_y = pt[1] + int(k * dist * self.norms[idx][1])
                    t.append((new_x, new_y))
                    new_pts[idx] = (new_x, new_y)
                else:
                    t.append(pt)
            morphTriangle(img1, imgMorphNorm, t1, t, cv2.INTER_CUBIC)
            new_triangles.append(t)
        self.pts = new_pts
        return imgMorphNorm.clip(0,255).astype(np.uint8), new_triangles
        
    def eyebrows(self, img, triangles, k):
        new_pts = self.pts.copy()
        k = k / 10.
        img1 = np.float32(img.copy())
        imgMorphNorm = np.zeros(img1.shape, dtype = img1.dtype)
        new_triangles = []
        for t1 in triangles:
            t = []
            for pt in t1:
                idx = pts_ind(self.pts, pt)
                if idx in range(22, 26):
                    dist = 0.7 * sqrt((self.pts[44][0] - self.pts[24][0])**2 + (self.pts[44][1] - self.pts[24][1])**2)
                    new_x = pt[0] + int(k * dist * self.norms[idx][0])
                    new_y = pt[1] + int(k * dist * self.norms[idx][1])
                    t.append((new_x, new_y))
                    new_pts[idx] = (new_x, new_y)
                elif idx in range(18, 22):
                    dist = 0.7 * sqrt((self.pts[37][0] - self.pts[19][0])**2 + (self.pts[37][1] - self.pts[19][1])**2)
                    new_x = pt[0] + int(k * dist * self.norms[idx][0])
                    new_y = pt[1] + int(k * dist * self.norms[idx][1])
                    t.append((new_x, new_y))
                    new_pts[idx] = (new_x, new_y)
                else:
                    t.append(pt)
            morphTriangle(img1, imgMorphNorm, t1, t, cv2.INTER_CUBIC)
            new_triangles.append(t)
        self.pts = new_pts
        return imgMorphNorm.clip(0,255).astype(np.uint8), new_triangles
        
    def nose_length(self, img, triangles, k):
        new_pts = self.pts.copy()
        k = k / 10.
        img1 = np.float32(img.copy())
        imgMorphNorm = np.zeros(img1.shape, dtype = img1.dtype)
        new_triangles = []
        dist = 0.5 * sqrt((self.pts[51][0] - self.pts[33][0])**2 + (self.pts[51][1] - self.pts[33][1])**2)
        for t1 in triangles:
            t = []
            for pt in t1:
                idx = pts_ind(self.pts, pt)
                if idx in range(30, 36):
                    new_x = pt[0] + int(k * dist * self.norms[30][0])
                    new_y = pt[1] + int(k * dist * self.norms[30][1])
                    t.append((new_x, new_y))
                    new_pts[idx] = (new_x, new_y)
                else:
                    t.append(pt)
            morphTriangle(img1, imgMorphNorm, t1, t, cv2.INTER_CUBIC)
            new_triangles.append(t)
        self.pts = new_pts
        return imgMorphNorm.clip(0,255).astype(np.uint8), new_triangles
    
    def nose_width(self, img, triangles, k):
        new_pts = self.pts.copy()
        k = k / 10.
        img1 = np.float32(img.copy())
        imgMorphNorm = np.zeros(img1.shape, dtype = img1.dtype)
        new_triangles = []
        dist = 0.2 * sqrt((self.pts[31][0] - self.pts[35][0])**2 + (self.pts[31][1] - self.pts[35][1])**2)
        for t1 in triangles:
            t = []
            for pt in t1:
                idx = pts_ind(self.pts, pt)
                if idx in (31, 35):
                    new_x = pt[0] + int(k * dist * self.norms[idx][0])
                    new_y = pt[1] + int(k * dist * self.norms[idx][1])
                    t.append((new_x, new_y))
                    new_pts[idx] = (new_x, new_y)
                else:
                    t.append(pt)
            morphTriangle(img1, imgMorphNorm, t1, t, cv2.INTER_CUBIC)
            new_triangles.append(t)
        self.pts = new_pts
        return imgMorphNorm.clip(0,255).astype(np.uint8), new_triangles
    
    def lips_thicc(self, img, triangles, k):
        new_pts = self.pts.copy()
        k = k / 10.
        img1 = np.float32(img.copy())
        imgMorphNorm = np.zeros(img1.shape, dtype = img1.dtype)
        new_triangles = []
        for t1 in triangles:
            t = []
            for pt in t1:
                idx = pts_ind(self.pts, pt)
                if idx in range(49, 54):
                    dist1 = sqrt((self.pts[50][0] - self.pts[61][0])**2 + (self.pts[50][1] - self.pts[61][1])**2)
                    dist2 = sqrt((self.pts[52][0] - self.pts[63][0])**2 + (self.pts[52][1] - self.pts[63][1])**2)
                    dist = 0.7 * (dist1 + dist2) / 2
                    new_x = pt[0] + int(k * dist * self.norms[idx][0])
                    new_y = pt[1] + int(k * dist * self.norms[idx][1])
                    t.append((new_x, new_y))
                    new_pts[idx] = (new_x, new_y)
                elif idx in range(55, 60):
                    dist = 0.3 * sqrt((self.pts[57][0]- self.pts[66][0])**2 + (self.pts[57][1] - self.pts[66][1])**2)
                    new_x = pt[0] + int(k * dist * self.norms[idx][0])
                    new_y = pt[1] + int(k * dist * self.norms[idx][1])
                    t.append((new_x, new_y))
                    new_pts[idx] = (new_x, new_y)
                else:
                    t.append(pt)
            morphTriangle(img1, imgMorphNorm, t1, t, cv2.INTER_CUBIC)
            new_triangles.append(t)
        self.pts = new_pts
        return imgMorphNorm.clip(0,255).astype(np.uint8), new_triangles
    
    def lips_width(self, img, triangles, k):
        new_pts = self.pts.copy()
        k = k / 10.
        img1 = np.float32(img.copy())
        imgMorphNorm = np.zeros(img1.shape, dtype = img1.dtype)
        new_triangles = []
        for t1 in triangles:
            t = []
            for pt in t1:
                idx = pts_ind(self.pts, pt)
                if idx == 48:
                    dist = 0.5 * sqrt((self.pts[48][0] - self.pts[4][0])**2 + (self.pts[48][1] - self.pts[4][1])**2)
                    new_x = pt[0] + int(k * dist * self.norms[48][0])
                    new_y = pt[1] + int(k * dist * self.norms[48][1])
                    t.append((new_x, new_y))
                    new_pts[idx] = (new_x, new_y)
                elif idx == 54:
                    dist = 0.5 * sqrt((self.pts[12][0] - self.pts[54][0])**2 + (self.pts[12][1] - self.pts[54][1])**2)
                    new_x = pt[0] + int(k * dist * self.norms[54][0])
                    new_y = pt[1] + int(k * dist * self.norms[54][1])
                    t.append((new_x, new_y))
                    new_pts[idx] = (new_x, new_y)
                else:
                    t.append(pt)
            morphTriangle(img1, imgMorphNorm, t1, t, cv2.INTER_CUBIC)
            new_triangles.append(t)
        self.pts = new_pts
        return imgMorphNorm.clip(0,255).astype(np.uint8), new_triangles
    
    def smile_on(self, img, triangles, k):
        new_pts = self.pts.copy()
        k = k / 10.
        img1 = np.float32(img.copy())
        imgMorphNorm = np.zeros(img1.shape, dtype = img1.dtype)
        new_triangles = []
        for t1 in triangles:
            t = []
            for pt in t1:
                idx = pts_ind(self.pts, pt)
                if idx in (48, 49, 59):
                    dist = 0.5 * sqrt((self.pts[51][0] - self.pts[33][0])**2 + (self.pts[51][1] - self.pts[33][1])**2)
                    drct = (self.norms[48][0] + self.norms[49][0], self.norms[48][1] + self.norms[49][1])
                    drctlen = sqrt(drct[0]**2 + drct[1]**2)
                    drct = (drct[0] / drctlen, drct[1] / drctlen)
                    new_x = pt[0] + int(k * dist * drct[0])
                    new_y = pt[1] + int(k * dist * drct[1])
                    if idx == 48:
                        new_x += int(2*k * dist * drct[0])
                        new_y += int(2*k * dist * drct[1])
                    t.append((new_x, new_y))
                    new_pts[idx] = (new_x, new_y)
                elif idx in (53, 54, 55):
                    dist = 0.5 * sqrt((self.pts[51][0] - self.pts[33][0])**2 + (self.pts[51][1] - self.pts[33][1])**2)
                    drct = (self.norms[53][0] + self.norms[54][0], self.norms[53][1] + self.norms[54][1])
                    drctlen = sqrt(drct[0]**2 + drct[1]**2)
                    drct = (drct[0] / drctlen, drct[1] / drctlen)
                    new_x = pt[0] + int(k * dist * drct[0])
                    new_y = pt[1] + int(k * dist * drct[1])
                    if idx == 54:
                        new_x += int(2*k * dist * drct[0])
                        new_y += int(2*k * dist * drct[1])
                    t.append((new_x, new_y))
                    new_pts[idx] = (new_x, new_y)
                else:
                    t.append(pt)
            morphTriangle(img1, imgMorphNorm, t1, t, cv2.INTER_CUBIC)
            new_triangles.append(t)
        self.pts = new_pts
        return imgMorphNorm.clip(0,255).astype(np.uint8), new_triangles
    
    def skull_width(self, img, triangles, k):
        new_pts = self.pts.copy()
        k = k / 10.
        img1 = np.float32(img.copy())
        imgMorphNorm = np.zeros(img1.shape, dtype = img1.dtype)
        new_triangles = []
        for t1 in triangles:
            t = []
            for pt in t1:
                idx = pts_ind(self.pts, pt)
                if idx == 1:
                    dist = 0.3 * sqrt((self.pts[1][0] - self.pts[36][0])**2 + (self.pts[1][1] - self.pts[36][1])**2)
                    new_x = pt[0] + int(k * dist * self.norms[idx][0])
                    new_y = pt[1] + int(k * dist * self.norms[idx][1])
                    t.append((new_x, new_y))
                    new_pts[idx] = (new_x, new_y)
                elif idx == 15:
                    dist = 0.3 * sqrt((self.pts[15][0] - self.pts[45][0])**2 + (self.pts[15][1] - self.pts[45][1])**2)
                    new_x = pt[0] + int(k * dist * self.norms[idx][0])
                    new_y = pt[1] + int(k * dist * self.norms[idx][1])
                    t.append((new_x, new_y))
                    new_pts[idx] = (new_x, new_y)
                else:
                    t.append(pt)
            morphTriangle(img1, imgMorphNorm, t1, t, cv2.INTER_CUBIC)
            new_triangles.append(t)
        self.pts = new_pts
        return imgMorphNorm.clip(0,255).astype(np.uint8), new_triangles
        
    def jaw_width(self, img, triangles, k):
        new_pts = self.pts.copy()
        k = k / 10.
        img1 = np.float32(img.copy())
        imgMorphNorm = np.zeros(img1.shape, dtype = img1.dtype)
        new_triangles = []
        for t1 in triangles:
            t = []
            for pt in t1:
                idx = pts_ind(self.pts, pt)
                if idx == 4:
                    dist = 0.4 * sqrt((self.pts[4][0] - self.pts[48][0])**2 + (self.pts[4][1] - self.pts[48][1])**2)
                    new_x = pt[0] + int(k * dist * self.norms[idx][0])
                    new_y = pt[1] + int(k * dist * self.norms[idx][1])
                    t.append((new_x, new_y))
                    new_pts[idx] = (new_x, new_y)
                elif idx == 12:
                    dist = 0.4 * sqrt((self.pts[12][0] - self.pts[54][0])**2 + (self.pts[12][1] - self.pts[54][1])**2)
                    new_x = pt[0] + int(k * dist * self.norms[idx][0])
                    new_y = pt[1] + int(k * dist * self.norms[idx][1])
                    t.append((new_x, new_y))
                    new_pts[idx] = (new_x, new_y)
                else:
                    t.append(pt)
            morphTriangle(img1, imgMorphNorm, t1, t, cv2.INTER_CUBIC)
            new_triangles.append(t)
        self.pts = new_pts
        return imgMorphNorm.clip(0,255).astype(np.uint8), new_triangles
        
    def chin_height(self, img, triangles, k):
        new_pts = self.pts.copy()
        k = k / 10.
        img1 = np.float32(img.copy())
        imgMorphNorm = np.zeros(img1.shape, dtype = img1.dtype)
        new_triangles = []
        for t1 in triangles:
            t = []
            for pt in t1:
                idx = pts_ind(self.pts, pt)
                if idx in (7,9):
                    dist = 0.5 * sqrt((self.pts[8][0] - self.pts[57][0])**2 + (self.pts[8][1] - self.pts[57][1])**2)
                    new_x = pt[0] + int(k * dist * self.norms[8][0])
                    new_y = pt[1] + int(k * dist * self.norms[8][1])
                    t.append((new_x, new_y))
                    new_pts[idx] = (new_x, new_y)
                else:
                    t.append(pt)
            morphTriangle(img1, imgMorphNorm, t1, t, cv2.INTER_CUBIC)
            new_triangles.append(t)
        self.pts = new_pts
        return imgMorphNorm.clip(0,255).astype(np.uint8), new_triangles
        
    def forehead_height(self, img, triangles, k):
        new_pts = self.pts.copy()
        k = k / 10.
        img1 = np.float32(img.copy())
        imgMorphNorm = np.zeros(img1.shape, dtype = img1.dtype)
        new_triangles = []
        for t1 in triangles:
            t = []
            for pt in t1:
                idx = pts_ind(self.pts, pt)
                if idx in range(69, 74):
                    dist1 = sqrt((self.pts[70][0] - self.pts[19][0])**2 + (self.pts[70][1] - self.pts[19][1])**2)
                    dist2 = sqrt((self.pts[72][0] - self.pts[24][0])**2 + (self.pts[72][1] - self.pts[24][1])**2)
                    dist = 0.3 * (dist1 + dist2) / 2
                    new_x = pt[0] + int(k * dist * norms[idx][0])
                    new_y = pt[1] + int(k * dist * norms[idx][1])
                    t.append((new_x, new_y))
                else:
                    t.append(pt)
            morphTriangle(img1, imgMorphNorm, t1, t, cv2.INTER_CUBIC)
            new_triangles.append(t)
        self.pts = new_pts
        return imgMorphNorm.clip(0,255).astype(np.uint8), new_triangles        

def make_face_triang(pts_src, size):
    rect = (0, 0, size[1], size[0])
    subdiv = cv2.Subdiv2D(rect)
    pts = []
    for pt in pts_src:
        pts.append(tuple(pt))
    for pt in pts:
        subdiv.insert(pt)
    triang4mask = subdiv.getTriangleList()
    return triang4mask            

def get_triang_mask(shape, triang4mask):
    matte = np.zeros(shape, dtype=np.float32)
    for tr in triang4mask:
        x1, y1, x2, y2, x3, y3 = tr
        t = [(x1, y1), (x2, y2), (x3, y3)]
        r = cv2.boundingRect(np.float32([t]))
        cv2.fillConvexPoly(matte, np.int32(t), (1.0, 1.0, 1.0), 16, 0)
    return matte            

def get_nn_mask(img, model):
    h, w, d = img.shape
    bbox = get_face_bbox(img)
    if bbox is not None:
        x1, y1, x2, y2 = bbox
        im = img[max(0,y1):min(y2,h), max(0,x1):min(x2,w)]
    inp_im = vgg_preprocess(im)
    out = model.predict([inp_im])
    out_resized = cv2.resize(np.squeeze(out), (im.shape[1],im.shape[0]))
    out_resized_clipped = np.clip(out_resized.argmax(axis=2), 0, 1).astype(np.float64)
    mask = np.zeros((h,w,1))
    mask[max(0,y1):min(y2,h), max(0,x1):min(x2,w),:] = out_resized_clipped[:,:,np.newaxis]
    return mask


def get_img_parts_masks(shape, pts_src):
    rect = (0, 0, shape[1], shape[0])
    subdiv = cv2.Subdiv2D(rect)
    pts = []
    for pt in pts_src:
        pts.append(tuple(pt))
    for i in range(36,42):
        subdiv.insert(pts[i])
    triang4leye = subdiv.getTriangleList()
    leye_mask = np.zeros(shape, dtype=np.float32)
    for tr in triang4leye:
        x1, y1, x2, y2, x3, y3 = tr
        t = [(x1, y1), (x2, y2), (x3, y3)]
        r = cv2.boundingRect(np.float32([t]))
        cv2.fillConvexPoly(leye_mask, np.int32(t), (1.0, 1.0, 1.0), 16, 0)

    subdiv = cv2.Subdiv2D(rect)
    pts = []
    for pt in pts_src:
        pts.append(tuple(pt))
    for i in range(42,48):
        subdiv.insert(pts[i])
    triang4reye = subdiv.getTriangleList()
    reye_mask = np.zeros(shape, dtype=np.float32)
    for tr in triang4reye:
        x1, y1, x2, y2, x3, y3 = tr
        t = [(x1, y1), (x2, y2), (x3, y3)]
        r = cv2.boundingRect(np.float32([t]))
        cv2.fillConvexPoly(reye_mask, np.int32(t), (1.0, 1.0, 1.0), 16, 0)

    subdiv = cv2.Subdiv2D(rect)
    pts = []
    for pt in pts_src:
        pts.append(tuple(pt))
    for i in lips_left_idx:
        subdiv.insert(pts[i])
    triang4llips = subdiv.getTriangleList()
    llips_mask = np.zeros(shape, dtype=np.float32)
    for tr in triang4llips:
        x1, y1, x2, y2, x3, y3 = tr
        t = [(x1, y1), (x2, y2), (x3, y3)]
        r = cv2.boundingRect(np.float32([t]))
        cv2.fillConvexPoly(llips_mask, np.int32(t), (1.0, 1.0, 1.0), 16, 0)

    subdiv = cv2.Subdiv2D(rect)
    pts = []
    for pt in pts_src:
        pts.append(tuple(pt))
    for i in lips_right_idx:
        subdiv.insert(pts[i])
    triang4rlips = subdiv.getTriangleList()
    rlips_mask = np.zeros(shape, dtype=np.float32)
    for tr in triang4rlips:
        x1, y1, x2, y2, x3, y3 = tr
        t = [(x1, y1), (x2, y2), (x3, y3)]
        r = cv2.boundingRect(np.float32([t]))
        cv2.fillConvexPoly(rlips_mask, np.int32(t), (1.0, 1.0, 1.0), 16, 0)

    eyes_mask = np.logical_or(leye_mask, reye_mask).astype(np.float32)
    lips_mask = np.logical_or(llips_mask, rlips_mask).astype(np.float32)
    eyeslips_mask = np.logical_or(eyes_mask, lips_mask).astype(np.float32)
    skin_mask = np.logical_and(face_mask, 1-eyeslips_mask).astype(np.float32)
    return eyes_mask, lips_mask, skin_mask


def makeup_preprocess(img):
    return (img / 255. - 0.5) * 2

def makeup_deprocess(img):
    return (img + 1) / 2


class Stylist:
    def __init__(self, face, skin, eyes, lips):
        self.face = face
        self.skin = skin
        self.eyes = eyes
        self.lips = lips
        self.smooth_mask = cv2.GaussianBlur(self.skin, (101,101), -1).clip(0,1)
        self.face_bl = cv2.GaussianBlur(face, (151,151), -1).clip(0,1)
        self.lips_bl = cv2.GaussianBlur(lips, (31,31), -1).clip(0,1)
        self.tone_mask = self.face_bl - cv2.GaussianBlur(eyes, (31,31), -1).clip(0,1) - self.lips_bl
    
    def smooth(self, img, k):
        face = (img * self.smooth_mask).astype(np.uint8)
        face_sm = cv2.bilateralFilter(face, 2*k, 50, 50)  
        img_sm = (img * (1-self.smooth_mask) + face_sm).clip(0,255).astype(np.uint8)
        return img_sm
    
    def skin_tone(self, img, k):
        img_hsv = rgb2hsv(img)
        img_hsv[:,:,1] -= 4*k /255
        img1 = hsv2rgb(img_hsv.clip(0,1))*255
        img_tone = (img * (1-self.tone_mask) + img1 * self.tone_mask).clip(0,255).astype(np.uint8)
        return img_tone
    
    def lips_tone(self, img, k):
        img_hsv = rgb2hsv(img)
        img_hsv[:,:,1] -= 4*k /255
        img1 = hsv2rgb(img_hsv.clip(0,1))*255
        img_tone = (img * (1-self.lips_bl) + img1 * self.lips_bl).clip(0,255).astype(np.uint8)
        return img_tone
    
    def contrast(self, img, k):
        img1 = np.float32(img.copy())
        facemask = self.face[:,:,0].astype(np.bool)
        faceH, faceW = faceHW(facemask)
        img_skcla = (equalize_adapthist(img, kernel_size=(faceH/5,faceW/3), clip_limit=0.001*k)*255).clip(0,255).astype(np.uint8)
        img_contr  = (img * (1-self.face_bl) + img_skcla * self.face_bl).clip(0,255).astype(np.uint8)
        return img_contr
        
    def makeup(self, img_full, k):
        #~ K.clear_session()
        H, W, _ = img_full.shape
        bbox = get_face_bbox(img_full)
        if bbox is not None:
            x1, y1, x2, y2 = bbox
            no_makeup = img_full[max(0,y1):min(y2, H), max(0,x1):min(x2, W)]
            bbH, bbW, _ = no_makeup.shape
        img_full = img_full / 255.
        no_makeup = cv2.resize(no_makeup, (256, 256))
        X_img = np.expand_dims(makeup_preprocess(no_makeup), 0)
        makeups = glob(os.path.join('../BeautyGAN/imgs', 'makeup', '*.*'))
        tf.reset_default_graph()
        sess = tf.Session()
        sess.run(tf.global_variables_initializer())
        saver = tf.train.import_meta_graph(os.path.join('../BeautyGAN/model', 'model.meta'))
        saver.restore(sess, tf.train.latest_checkpoint('../BeautyGAN/model'))
        graph = tf.get_default_graph()
        X = graph.get_tensor_by_name('X:0')
        Y = graph.get_tensor_by_name('Y:0')
        Xs = graph.get_tensor_by_name('generator/xs:0')
        makeup = cv2.resize(io.imread(makeups[k-1]), (256, 256))
        Y_img = np.expand_dims(makeup_preprocess(makeup), 0)
        Xs_ = sess.run(Xs, feed_dict={X: X_img, Y: Y_img})
        Xs_ = makeup_deprocess(Xs_)
        dif = (Xs_[0] - no_makeup / 255.)
        dif = cv2.resize(dif, (bbW, bbH))
        mask = np.zeros_like(img_full)
        mask[max(0,y1):min(y2, H), max(0,x1):min(x2, W)] = dif
        mask = mask * self.face_bl
        img_full += mask
        return np.uint8(img_full.clip(0,1) * 255)


#parsing cmdline arguments if needed
parser = argparse.ArgumentParser()
parser.add_argument('--input', type=str, default='00049.jpg', help='input photo')
parser.add_argument('--out', type=str, default='result.jpg', help='output file name')
args = parser.parse_args()

#initialize the demands
morph_demands = {'eyebrows': 0,
                 'eyes_size': 0,
                 'nose_length': 0,
                 'nose_width': 0,
                 'smile_on': 0,
                 'lips_thicc': 0,
                 'lips_width': 0,
                 'skull_width': 0,
                 'jaw_width': 0,
                 'chin_height': 0,
                 'forehead_height': 0}

skin_demands = {'makeup': 5,
                'contrast': 0,
                'skin_tone': 0,
                'lips_tone': 0,
                'smooth': 0}

#read the input image and get its shape
img = io.imread(args.input)
H, W, D = img.shape

#predict facial kepoints
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor('shape_predictor_68_face_landmarks.dat')
dets = detector(img, 1)
parts = predictor(img, dets[0]).parts()
dlib_pts = []
for pt in parts:
    dlib_pts.append(np.array([pt.x, pt.y]))
dlib_pts = np.array(dlib_pts)

#mask from triangulation
face_triangles = make_face_triang(dlib_pts, (H,W))
triang_mask = get_triang_mask(img.shape, face_triangles)

# NN segmentation
seg_model = FCN()
seg_model.load_weights("Keras_FCN8s_face_seg_YuvalNirkin.h5")
nn_mask = get_nn_mask(img, seg_model)
K.clear_session()

#get hairline from NN mask
contour = find_contours(nn_mask[:,:,0], 0.5)[0].astype(np.int16)
contour = contour[np.logical_or(np.logical_and(contour[:,0] < dlib_pts[0][1], np.abs(contour[:,1] - dlib_pts[0][0]) <= np.abs(contour[:,1] - dlib_pts[16][0])),
                                np.logical_and(contour[:,0] < dlib_pts[16][1], np.abs(contour[:,1] - dlib_pts[0][0]) > np.abs(contour[:,1] - dlib_pts[16][0])))]
contour = contour[np.linspace(0, len(contour)-1, num=9, endpoint=True).astype(np.uint16)][1:-1]
#add hairline to other pts
dlib_pts = np.concatenate([dlib_pts, contour[:,::-1]], axis=0)

#get normals for points
norms = get_normals(dlib_pts)

#get triangulation for morphing
morph_triangles, pts = img_sparse_triang(dlib_pts, (H, W))
triangles = []
for tr1 in morph_triangles:
    x1, y1, x2, y2, x3, y3 = tr1
    t1 = [(x1, y1), (x2, y2), (x3, y3)]
    triangles.append(t1)

#get working masks
face_mask = np.logical_or(triang_mask, nn_mask).astype(np.float32)
eyes_mask, lips_mask, skin_mask = get_img_parts_masks(img.shape, pts) #was morpher.pts

#create beauty crew
morpher = Morpher(norms, pts)
stylist = Stylist(face_mask, skin_mask, eyes_mask, lips_mask)

#process tone step-by-step
for demand in skin_demands.items():
    if demand[1] != 0:
        img = getattr(stylist, demand[0])(img, demand[1])

#morph step-by-step better after stylist
for demand in morph_demands.items():
    if demand[1] != 0:
        img, triangles = getattr(morpher, demand[0])(img, triangles, demand[1]) 

#print the result
cv2.imwrite(args.out, np.flip(img, 2))




























