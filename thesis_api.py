#!python3
import argparse
import os
import torch
from datetime import timedelta, datetime
from functools import update_wrapper, wraps
import cv2
import settings
import io

from PIL import Image
from PIL import ImageFont, ImageDraw
import traceback
import sys
import cv2 as cv
import numpy as np
import json
import logging
import traceback
import cv2
import io
import os
import pickle
import time
from logging.handlers import RotatingFileHandler, TimedRotatingFileHandler
from datetime import timedelta, datetime
from functools import update_wrapper, wraps
import time
from flask import Flask, Response, request, render_template, make_response, current_app
import argparse
import settings
from threading import Thread
import requests
import uuid
import logging
from logging.handlers import RotatingFileHandler, TimedRotatingFileHandler
import shutil
import numpy as np
from experiment import Structure, Experiment
from concern.config import Configurable, Config
import math
from flask import Flask, Response, request, render_template, make_response, current_app
def cross_domain(origin=None, methods=None, headers=None,
                 max_age=21600, attach_to_all=True,
                 automatic_options=True):
    if methods is not None:
        methods = ', '.join(sorted(x.upper() for x in methods))
    if headers is not None and not isinstance(headers, str):
        headers = ', '.join(x.upper() for x in headers)
    if not isinstance(origin, str):
        origin = ', '.join(origin)
    if isinstance(max_age, timedelta):
        max_age = max_age.total_seconds()

    def get_methods():
        if methods is not None:
            return methods

        options_resp = current_app.make_default_options_response()
        return options_resp.headers['allow']

    def decorator(f):
        def wrapped_function(*args, **kwargs):
            if automatic_options and request.method == 'OPTIONS':
                resp = current_app.make_default_options_response()
            else:
                resp = make_response(f(*args, **kwargs))
            if not attach_to_all and request.method != 'OPTIONS':
                return resp

            h = resp.headers

            h['Access-Control-Allow-Origin'] = origin
            h['Access-Control-Allow-Methods'] = get_methods()
            h['Access-Control-Max-Age'] = str(max_age)
            if headers is not None:
                h['Access-Control-Allow-Headers'] = headers
            return resp

        f.provide_automatic_options = False
        return update_wrapper(wrapped_function, f)

    return decorator


def shrink_image_func(binary_img):
    
    histogram_x=np.sum(binary_img,axis=0)
    delta_x0=0
    for i in range(int(histogram_x.shape[0]/2)):
        if histogram_x[i]==0:
            delta_x0=i
        else:
            break
    
    
    
    delta_x3=histogram_x.shape[0]-1
    for i in range(histogram_x.shape[0]-1,int(histogram_x.shape[0]/2),-1):
        if histogram_x[i]==0:
            delta_x3=i
        else:
            break
    
    
    
    
    histogram_y=np.sum(binary_img,axis=1)
    
    delta_y0=0
    for i in range(int(histogram_y.shape[0]/2)):
        if histogram_y[i]==0:
            delta_y0=i
        else:
            break
    
    
    delta_y3=histogram_y.shape[0]-1
    for i in range(histogram_y.shape[0]-1,int(histogram_y.shape[0]/2),-1):
        if histogram_y[i]==0:
            delta_y3=i
        else:
            break
    
    
    delta_x0=max(0,delta_x0-1)
    delta_y0=max(0,delta_y0-1)
    delta_x3=min(binary_img.shape[1],delta_x3+1)
    delta_y3=min(binary_img.shape[0],delta_y3+1)
    return binary_img[delta_y0:delta_y3,delta_x0:delta_x3],[(delta_x0,delta_y0),(delta_x3,delta_y3)]
   
    croped_image=img[p1[1]:p3[1],p1[0]:p3[0]]

def split_long_box_func(image,id):
    box=[0,0,image.shape[1],image.shape[0]]
    
    newbox=box
    results = []
    boxes = [newbox]
    im_box=image
    space=[0]*im_box.shape[1]
    for i in range(im_box.shape[1]):
        #print(im_box[:,i])
        if all([a==0 for a in im_box[:,i]]):
            space[i]=1
    
    for window_size in range(30,1,-1):
        
        #window_size=int(image.shape[0])*(100-ratio)/100
        space_line=[]
        count=0
        split_line=[]
        used=[0]*image.shape[1]
        for i in range(image.shape[1]):
            if used[i]==1:
                continue
            for j in range(i+1,image.shape[1]):
                if all([space[k]==1 for k in range(i,j)]):
                    count=j-i
                else:
                    break
            
            if count>=window_size:
                split_line.append((i,i+count))
                for k in range(i,i+count):
                    used[k]=1
                count=0
       
        split_line.insert(0,(0,0))
        split_line.append((image.shape[1],image.shape[1]))
        
        splited_images=[]
       
        if True:
            start=0
            
            while True:
                crop_img=None
                next_step=-1
                for count_i in range(len(split_line)-1,start,-1):
                    crop_img=image[:,split_line[start][1]:split_line[count_i][0]]
                    
                    next_step=count_i
                    if crop_img.shape[1]/crop_img.shape[0]>6:
                        continue
                    
                    break
                if crop_img.shape[1]>4:
                    splited_images.append(crop_img)
                
                start=next_step
                
                if start>=len(split_line)-1:
                    break
        else:
            for i in range(len(split_line)-1):
                if abs(split_line[i][1]- split_line[i+1][0])<4:
                    continue
                splited_images.append(image[:,split_line[i][1]:split_line[i+1][0]])
        
      
        if all(splited_images[i].shape[1]/splited_images[i].shape[0]<6 for i in range(len(splited_images))):
            break
        if id==21:
            print(window_size,len(splited_images))
    return splited_images
app = Flask(__name__, static_folder='static')
formatter = logging.Formatter('%(asctime)s %(levelname)s %(message)s')
handler = TimedRotatingFileHandler('log/server_thesis.log', when="midnight", interval=1, backupCount=20)
handler.setFormatter(formatter)
app.logger.addHandler(handler)
app.logger.setLevel(settings.LOG_LEVEL)

debug={"crop_img":True,"binary_img":True,"recognition":True,"shrink":True,"split_long_box":True}
@app.route('/thesis', methods=['POST'])
@cross_domain(origin='*', headers='Content-Type')
def content_post():
    try:
        ffile = request.files['file']
        bio = io.BytesIO()
        ffile.save(bio)
        fname = os.path.basename(ffile.filename)
        image = cv.imdecode(np.frombuffer(bio.getvalue(), dtype='uint8'), 1)
        
        cv2.imwrite("debug/thesis_input/"+fname,image)
        
        
        
        # =========== detection ==============
        with open("debug/thesis_input/"+fname,'rb') as f:
            detection_result  = requests.post(url="http://localhost:6001/api/post/ocr", files={'file':f})
            detection_result = detection_result.json()
        
        
        
        recognition_images=[]
        # ============ crop image ==============
        for polygon_data in detection_result['data']:
            
            poly=polygon_data[0]
            score=polygon_data[1]
            print(poly)
            rotrect = cv2.minAreaRect(np.array(poly,dtype=np.int32))
            box_rec = cv2.boxPoints(rotrect)
            recognition_images.append({"box_rec":box_rec,"image":image[int(min(box_rec[:,1])):int(max(box_rec[:,1])),
                                                                        int(min(box_rec[:,0])):int(max(box_rec[:,0]))]})
           
        
        if debug['crop_img']:
        # ============ crop image output debug ==============
            for crop_id,croped_img_data in enumerate(recognition_images):
                crop_img=croped_img_data['image']
                
                cv2.imwrite("debug/crop_img/"+str(crop_id)+".jpg",crop_img)
                
        # ============ binary image ==============
        for crop_id,croped_img_data in enumerate(recognition_images):
            crop_img=croped_img_data['image']
            croped_image_gray = cv2.cvtColor(crop_img, cv2.COLOR_BGR2GRAY)  
            (thresh, im_bw) = cv2.threshold(croped_image_gray, 127, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)
            croped_img_data['binary_img']=im_bw
        if debug['binary_img']:
        # ============ binary image output debug ==============
            for crop_id,croped_img_data in enumerate(recognition_images):
                binary_img=croped_img_data['binary_img']
                cv2.imwrite("debug/crop_img/"+str(crop_id)+"_binary.jpg",binary_img)
        
        
        #============= shrink box ==============
        for crop_id,croped_img_data in enumerate(recognition_images):
            croped_img_data['shrink_img']=shrink_image_func(croped_img_data['binary_img'])
        if debug['shrink']:    
            # ============ binary image output debug ==============
            for crop_id,croped_img_data in enumerate(recognition_images):
                shrink_img=croped_img_data['shrink_img'][0]
                cv2.imwrite("debug/crop_img/"+str(crop_id)+"_shrink_img.jpg",shrink_img)
        
        #============= split long box ==========
        for crop_id,croped_img_data in enumerate(recognition_images):
            croped_img_data['split_long_box']=split_long_box_func(croped_img_data['shrink_img'][0],crop_id)
        
        if debug['split_long_box']:
        # ============ split long box debug ==============
            for crop_id,croped_img_data in enumerate(recognition_images):
                
                for split_img_i,split_img in enumerate(croped_img_data['split_long_box']):
                    cv2.imwrite("debug/split_longbox/"+str(crop_id)+"_"+str(split_img_i)+"_split.jpg",split_img)
        # ============ recongition OCR mix ==============
        
        for crop_id,croped_img_data in enumerate(recognition_images):
            croped_img_data['split_long_text']=[]
            for split_img_i,split_img in enumerate(croped_img_data['split_long_box']):
                with open("debug/split_longbox/"+str(crop_id)+"_"+str(split_img_i)+"_split.jpg", 'rb') as f:      
                    r = requests.post(url="http://192.168.0.124:8512/aster", files={'file': f},data={})
                    croped_img_data['split_long_text'].append(r.json())
            
        if debug['recognition'] or True:
        # ============ visualize OCR mix ==============
            
            
            #ocr_images=Image.new("RGB",[image.shape[1],image.shape[0]])
            ocr_images = Image.fromarray(cv2.imread("debug/thesis_input/"+fname))
            draw = ImageDraw.Draw(ocr_images)
            for crop_id,croped_img_data in enumerate(recognition_images):
                text="".join(text['predict'][0]  for text in croped_img_data['split_long_text'])
                
                box_rec=np.array(croped_img_data['box_rec'])
                
                font=ImageFont.truetype("SIMSUN.ttf",40)
                #draw.text((max(0,min(box_rec[:,0]-60)),min(box_rec[:,1])), str(crop_id), font=font,fill=(0,0,255,128))
                #draw.text((min(box_rec[:,0]),min(box_rec[:,1])), text, font=font,fill=(122,8,44,128))
                
            ocr_images.save("debug/demo_results/ocr_images.png")
            ocr_images.save("/home/asilla/10_dataset/hunglh/thesis_demo/data/result_"+fname)
            ocr_images=cv2.imread("/home/asilla/10_dataset/hunglh/thesis_demo/data/result_"+fname)
            for crop_id,croped_img_data in enumerate(recognition_images):
                text="".join(text['predict'][0]  for text in croped_img_data['split_long_text'])
                
                box_rec=np.array(croped_img_data['box_rec'])
                
                font=ImageFont.truetype("SIMSUN.ttf",40)
                ocr_images = cv2.rectangle(ocr_images, (min(box_rec[:,0]),min(box_rec[:,1])), (max(box_rec[:,0]),max(box_rec[:,1])), (35,209,96), 2) 
            cv2.imwrite("/home/asilla/10_dataset/hunglh/thesis_demo/data/result_"+fname,ocr_images)
            
            with open("/home/asilla/10_dataset/hunglh/thesis_demo/data/result_"+fname+".pck","wb") as outfile:
                pickle.dump(recognition_images,outfile)
            # with open("/home/asilla/10_dataset/hunglh/thesis_demo/data/result_"+fname+".json","w",encoding="utf-8"):
                # json.dump({}
        
        status_code=200
        response = {
            'status': 'OK',
            'message': 'Unexpected error.'
        }
    except  Exception as ex:
        app.logger.error(ex)
        app.logger.error(str(traceback.format_exc()))
        status_code = 500

        response = {
            'status': 'ERROR',
            'message': 'Unexpected error.'
        }
    return Response(json.dumps(response), status=status_code, mimetype='application/json; charset=utf-8')    

def main():
    # parser = argparse.ArgumentParser()
    # parser.add_argument('--port', default=8769, type=int)
    # #parser.add_argument('--debug', action='store_true')
    # args = parser.parse_args()

    app.debug = True
    app.run('0.0.0.0', 6006)

if __name__ == '__main__':
    main()
