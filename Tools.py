#-*- coding: utf-8 -*-
import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties
import seaborn as sns
import os

class Tools:
    # 生成绘图用的颜色
    def rgbToHex(self,rgb):

        RGB = list(rgb)
        color = '#'
        for i in RGB:
            num = int(i)
            color += str(hex(num))[-2:].replace('x', '0').upper()
        return color

    def generate_colors_c(self,N=7,colormap='terrain'):
        step = max(int(255/N),1)
        cmap = plt.get_cmap(colormap)
        hex_list = []
        for i in range(N):
            id = step*i
            id = 255 if id>255 else id
            rgba_color = cmap(id)
            rgb = [int(d*255) for d in rgba_color[:3]]
            hex_list.append(self.rgbToHex(rgb))
        return hex_list

    def generate_colors(self,N=7,colormap='Set3'):
        cmap = plt.get_cmap(colormap)
        hex_list = []
        for i in range(N):
            i = 255 if i>255 else i
            rgba_color = cmap(i)
            rgb = [int(d*255) for d in rgba_color[:3]]
            hex_list.append(self.rgbToHex(rgb))
        return hex_list