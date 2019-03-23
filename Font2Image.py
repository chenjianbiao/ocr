#! /usr/bin/env python
# -*- coding: utf-8 -*-

import cv2
import numpy as np
from PIL import Image,ImageDraw,ImageFont



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
