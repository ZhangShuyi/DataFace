# -*- coding: utf-8 -*-
"""
    Version 1.0
    To fetch picture from Internet
    Use data from ModelAndTxt/FacesToBeFetched.txt
"""

import os
import re
import requests


FACE_TXT_FILE = "ModelAndTxt/FacesToBeFetched.txt"
PIC_URL_LIST = []


def getPictureFromInternet(txt_path):
    file_object = open(txt_path)
    i = 1
    readnum = 0
    startpos = 2691
    try:
        while (1):
            line = file_object.readline()
            readnum += 1
            if readnum < startpos:
                continue
            print("line:", readnum)
            line_s = line.split()
            if len(line_s) > 3 and re.match(r'^https?:/{2}\w.+$', line_s[-3]):
                print(line_s[-3], line_s[-4])
                try:
                    pic = requests.get(line_s[-3], timeout=20)
                    path_string = 'datas\\'
                    if len(line_s) == 6:
                        path_string = path_string + line_s[-6] + line_s[-5]
                    elif len(line_s) == 5:
                        path_string = path_string + line_s[-5]
                    else:
                        print("Info: Address is inValid")
                        print(path_string)
                        continue

                    if not os.path.exists(path_string):
                        os.mkdir(path_string)
                        i = 1
                    fp = open(path_string + '\\' + line_s[-4] + '.jpg', 'wb')
                    fp.write(pic.content)
                    fp.close()
                except requests.exceptions.ConnectionError:
                    print('【错误】当前图片无法下载')
                    continue
                except requests.exceptions.TooManyRedirects:
                    print('【错误】重定向太多')
                    continue

    finally:
        file_object.close()
