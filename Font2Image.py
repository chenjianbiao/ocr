#! /usr/bin/env python
# -*- coding: utf-8 -*-

import cv2
import numpy as np
from PIL import Image,ImageDraw,ImageFont
import FindImageBBox
import PreprocessResizeKeepRatioFillBG
import PIL
import PIL.F


class Font2Image(object):
    def __init__(self,width,hight,need_crop,margin):
        self.width=width
        self.hight=hight
        self.need_crop=need_crop
        self.margin=margin
    def do(self,font_path,char,rotate=0):
        find_image_bbox=FindImageBBox()
        img=Image.new('RGB',(self.width,self.hight),self.margin)
        draw=ImageDraw.Draw(img)
        font=ImageFont.truetype(font_path,int(self.width*0.7),)
        draw.text((0,0),char,(255,255,2555),font)
        if rotate!=0:
            img.rotate(rotate)
        data=list(img.getdata())
        sum_val=0
        for i_data in data:
            sum_val+=sum(i_data)
        if sum_val > 2:
            np_img=np.asarray(data,dtype='uint8')
            np_img=np_img[:,0]
            np_img=np_img.reshape((self.width,self.hight))
            cropped_box=find_image_bbox.do(np_img)
            left,upper,right,lower=cropped_box
            np_img=np_img[upper:lower+1,left:right+1]
            if not self.need_crop:
                preprocess_resize_keep_ratio_fill_bg=PreprocessResizeKeepRatioFillBG();
                np_img=preprocess_resize_keep_ratio_fill_bg.do(np_img)
                return np_img
        else:
            print("image doesn't exist.")




