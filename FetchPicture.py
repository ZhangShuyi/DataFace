# -*- coding: utf-8 -*-
"""
    Version 1.0
    To fetch picture from Internet
    Use data from ModelAndTxt/FacesToBeFetched.txt
"""
import os
import re
import requests
from InternalModule.LogSetting import ROOT_LOG, RECORD_LOG, PRINT_LOG

FACE_TXT_FILE = "ModelAndTxt/FacesToBeFetched.txt"
SAVE_PIC_PATH = "../DataFromInternet"
PIC_URL_LIST = []


def getPictureFromInternet(txt_path, save_path, start_line=0):
    ROOT_LOG.info(
        "Start Fetch the picture from Internet \n \
        \t According to {} (start line {}) save at {}".format(txt_path, start_line, save_path))
    try:
        file_object = open(txt_path)
    except FileNotFoundError:
        ROOT_LOG.error("NotFoundFile {}, the process was finished".format(txt_path))
        return
    index = 0
    total_picture = 0
    num_dict = {}
    while True:
        line = file_object.readline()
        if line is None:
            break
        index += 1
        if index < start_line:
            continue
        line_s = line.split()
        if len(line_s) > 3 and re.match(r'^https?:/{2}\w.+$', line_s[-3]):
            try:
                address = line_s[-3]
                pic = requests.get(address, timeout=20)
                if len(line_s) == 6:
                    name = line_s[-6] + line_s[-5]
                elif len(line_s) == 5:
                    name = line_s[-5]
                else:
                    PRINT_LOG.info("Line {} is invalid ({})".format(index, line_s))
                    continue
                people_save_file = os.path.join(save_path, name)
                if not os.path.exists(people_save_file):
                    ROOT_LOG.info("Generate file {}".format(people_save_file))
                    os.mkdir(people_save_file)
                if name not in num_dict:
                    num_dict[name] = 1
                else:
                    num_dict[name] += 1
                fp = open(os.path.join(people_save_file, "%04d" % num_dict[name] + '.jpg'), 'wb')
                fp.write(pic.content)
                total_picture += 1
                fp.close()
            except requests.exceptions.ConnectionError:
                PRINT_LOG.info("ConnectionError {}".format(address))
                continue
            except requests.exceptions.TooManyRedirects:
                PRINT_LOG.info("TooManyRedirects {}".format(address))
                continue
    ROOT_LOG.info(
        "End Fetch the picture from Internet \n \
        \t According to {}, success num {} save at {} ".format(txt_path, total_picture, save_path))
    file_object.close()


if __name__ == "__main__":
    if not os.path.exists(SAVE_PIC_PATH):
        os.mkdir(SAVE_PIC_PATH)
        ROOT_LOG.info("Generate file {}".format(SAVE_PIC_PATH))
    getPictureFromInternet(FACE_TXT_FILE, SAVE_PIC_PATH)
