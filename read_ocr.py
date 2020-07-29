# -*- coding: utf-8 -*-
# -.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.#

#* File Name : read_ocr.py
#
#* Purpose :
#
#* Creation Date : 06-07-2020
#
#* Last Modified : Saturday 11 July 2020 12:07:06 PM IST
#
#* Created By :

#_._._._._._._._._._._._._._._._._._._._._.#
try:
    from PIL import Image
except ImportError:
    import Image
import os
import pandas as pd

def main(ocrType):
    datadir = '../data/Dataset/'
    files = os.listdir(datadir)
    df = pd.DataFrame(columns=['fname', 'text'])
    if ocrType=='tesseract':
        import pytesseract
        for fil in files:
            image = Image.open(os.path.join(datadir,fil))
            binary_image = image.convert('L')
            string = pytesseract.image_to_string(binary_image)
            df = df.append({'fname': fil, 'text':string }, ignore_index=True)
    else:
        import easyocr
        reader = easyocr.Reader(['en'])
        for fil in files:
            #image = Image.open(os.path.join(datadir,fil))
            #binary_image = image.convert('L')
            string = reader.readtext(os.path.join(datadir, fil))
            df = df.append({'fname': fil, 'text':string }, ignore_index=True)
    return df

if __name__ == '__main__':
    df = main(ocrType='easyocr')
    df.to_csv('../data/%s_binary_extracted_text.csv'%ocrType, index=False)
