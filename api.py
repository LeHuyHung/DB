#!python3
import argparse
import os
import torch
from datetime import timedelta, datetime
from functools import update_wrapper, wraps
import cv2
import settings
import io
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

app = Flask(__name__, static_folder='static')
formatter = logging.Formatter('%(asctime)s %(levelname)s %(message)s')
handler = TimedRotatingFileHandler('log/server.log', when="midnight", interval=1, backupCount=20)
handler.setFormatter(formatter)
app.logger.addHandler(handler)
app.logger.setLevel(settings.LOG_LEVEL)
class Demo:
    def __init__(self, experiment, args, cmd=dict()):
        self.RGB_MEAN = np.array([122.67891434, 116.66876762, 104.00698793])
        self.experiment = experiment
        experiment.load('evaluation', **args)
        self.args = cmd
        model_saver = experiment.train.model_saver
        self.structure = experiment.structure
        self.model_path = self.args['resume']

    def init_torch_tensor(self):
        # Use gpu or not
        torch.set_default_tensor_type('torch.FloatTensor')
        if torch.cuda.is_available():
            self.device = torch.device('cuda')
            torch.set_default_tensor_type('torch.cuda.FloatTensor')
        else:
            self.device = torch.device('cpu')

    def init_model(self):
        model = self.structure.builder.build(self.device)
        return model

    def resume(self, model, path):
        if not os.path.exists(path):
            print("Checkpoint not found: " + path)
            return
        print("Resuming from " + path)
        states = torch.load(
            path, map_location=self.device)
        model.load_state_dict(states, strict=False)
        print("Resumed from " + path)

    def resize_image(self, img):
        height, width, _ = img.shape
        if height < width:
            new_height = self.args['image_short_side']
            new_width = int(math.ceil(new_height / height * width / 32) * 32)
        else:
            new_width = self.args['image_short_side']
            new_height = int(math.ceil(new_width / width * height / 32) * 32)
        resized_img = cv2.resize(img, (new_width, new_height))
        return resized_img
        
    def load_image(self, image_path):
        img = cv2.imread(image_path, cv2.IMREAD_COLOR).astype('float32')
        original_shape = img.shape[:2]
        img = self.resize_image(img)
        img -= self.RGB_MEAN
        img /= 255.
        img = torch.from_numpy(img).permute(2, 0, 1).float().unsqueeze(0)
        return img, original_shape
        
    def format_output(self, batch, output):
        batch_boxes, batch_scores = output
        
        formated_output=[]
        for index in range(batch['image'].size(0)):
            original_shape = batch['shape'][index]
            filename = batch['filename'][index]
            result_file_name = 'res_' + filename.split('/')[-1].split('.')[0] + '.txt'
            result_file_path = os.path.join(self.args['result_dir'], result_file_name)
            boxes = batch_boxes[index]
            scores = batch_scores[index]
            if self.args['polygon']:
                with open(result_file_path, 'wt') as res:
                    for i, box in enumerate(boxes):
                        print("polygon:",np.array(box).shape)
                        rotrect = cv2.minAreaRect(np.array(box))
                        box_poly=box
                        
                        box_rec = cv2.boxPoints(rotrect)
                        box_rec = np.int0(box_rec)
                        print(box_rec.shape)
                        box_rec=box_rec.reshape(-1).tolist()
                        box = np.array(box).reshape(-1).tolist()
                        result = ",".join([str(int(x)) for x in box])
                        score = scores[i]
                        res.write(result +"_"+ ",".join([str(int(x)) for x in box_rec])+'_' + str(score) + "\n")
                        formated_output.append([box_poly,score])
            else:
                with open(result_file_path, 'wt') as res:
                    for i in range(boxes.shape[0]):
                        score = scores[i]
                        if score < self.args['box_thresh']:
                            continue
                            
                        box_poly=boxes[i,:,:].tolist()
                        box = boxes[i,:,:].reshape(-1).tolist()
                        result = ",".join([str(int(x)) for x in box])
                        res.write(result + ',' + str(score) + "\n")
                        formated_output.append([box_poly,score])
        return formated_output
    def inference(self, image_path, visualize=False):
        
        self.init_torch_tensor()
        model = self.init_model()
        self.resume(model, self.model_path)
        model.eval()
        if os.path.isdir(image_path):
            image_folder=image_path
            formated_outputs=[]
            for image_path in os.listdir(image_folder):
                image_path=os.path.join(image_folder,image_path)
                print(image_path)
                all_matircs = {}
                
                batch = dict()
                batch['filename'] = [image_path]
                img, original_shape = self.load_image(image_path)
                batch['shape'] = [original_shape]
                with torch.no_grad():
                    batch['image'] = img
                    pred = model.forward(batch, training=False)
                    
                    dilation = cv2.dilate((pred.cpu().numpy()[0,0,:,:]>0.5).astype(np.uint8)*255,np.ones((5,5),np.uint8),iterations = 1)
                    dilation=np.array([[dilation]])
                    dilation=dilation/255/1.2
                    #================ end ======================
                    output = self.structure.representer.represent(batch, dilation, is_output_polygon=self.args['polygon']) 
                    if not os.path.isdir(self.args['result_dir']):
                        os.mkdir(self.args['result_dir'])
                    formated_output=self.format_output(batch, output)
                    formated_outputs.append(formated_output)
                    print(len(output[0][0]))
                    if True and self.structure.visualizer:
                        vis_image = self.structure.visualizer.demo_visualize(image_path, output)
                        cv2.imwrite(os.path.join(self.args['result_dir'], image_path.split('/')[-1].split('.')[0]+'.jpg'), vis_image)
                        predict_map=pred.cpu().numpy()[0,0,:,:]
                        predict_map=predict_map*254
                        predict_map=predict_map.astype(np.uint8)
                        predict_map=np.reshape(predict_map,(predict_map.shape[0],predict_map.shape[1],1))
                        cv2.imwrite(os.path.join(self.args['result_dir'], image_path.split('/')[-1].split('.')[0]+'_score.jpg'), predict_map)
                        # origin_img=cv2.imread(image_path)
                        # origin_img[:,:,0]=np.clip(origin_img[:,:,0]-binary, a_min = 0, a_max = 255) 
                        # origin_img[:,:,1]=np.clip(origin_img[:,:,1]-binary, a_min = 0, a_max = 255) 
                        # origin_img[:,:,2]=np.clip(origin_img[:,:,2]+binary, a_min = 0, a_max = 255) 
                        
            return formated_outputs
        else:
            
            all_matircs = {}
            batch = dict()
            batch['filename'] = [image_path]
            img, original_shape = self.load_image(image_path)
            batch['shape'] = [original_shape]
            with torch.no_grad():
                batch['image'] = img
                pred = model.forward(batch, training=False)
                dilation = cv2.dilate((pred.cpu().numpy()[0,0,:,:]>0.5).astype(np.uint8)*255,np.ones((5,5),np.uint8),iterations = 1)
                dilation=np.array([[dilation]])
                dilation=dilation/255/1.2
                output = self.structure.representer.represent(batch, dilation, is_output_polygon=self.args['polygon']) 
                if not os.path.isdir(self.args['result_dir']):
                    os.mkdir(self.args['result_dir'])
                formated_output=self.format_output(batch, output)

                if visualize and self.structure.visualizer:
                    vis_image = self.structure.visualizer.demo_visualize(image_path, output)
                    cv2.imwrite(os.path.join(self.args['result_dir'], image_path.split('/')[-1].split('.')[0]+'.jpg'), vis_image)
                    

            return formated_output
parser = argparse.ArgumentParser(description='Text Recognition Training')
parser.add_argument('exp', type=str)
parser.add_argument('--resume', type=str, help='Resume from checkpoint')
parser.add_argument('--port', default=8769, type=int)
parser.add_argument('--image_path', type=str, help='image path')
parser.add_argument('--result_dir', type=str, default='./demo_results/', help='path to save results')
parser.add_argument('--data', type=str,
                    help='The name of dataloader which will be evaluated on.')
parser.add_argument('--image_short_side', type=int, default=736,
                    help='The threshold to replace it in the representers')
parser.add_argument('--thresh', type=float,
                    help='The threshold to replace it in the representers')
parser.add_argument('--box_thresh', type=float, default=0.6,
                    help='The threshold to replace it in the representers')
parser.add_argument('--visualize', action='store_true',
                    help='visualize maps in tensorboard')
parser.add_argument('--resize', action='store_true',
                    help='resize')
parser.add_argument('--polygon', action='store_true',
                    help='output polygons if true')
parser.add_argument('--eager', '--eager_show', action='store_true', dest='eager_show',
                    help='Show iamges eagerly')

args = parser.parse_args()
args = vars(args)
args = {k: v for k, v in args.items() if v is not None}

conf = Config()
experiment_args = conf.compile(conf.load(args['exp']))['Experiment']
experiment_args.update(cmd=args)
experiment = Configurable.construct_class_from_config(experiment_args)

inference_endpoint=Demo(experiment, experiment_args, cmd=args)

@app.route('/api/post/ocr', methods=['POST'])
@cross_domain(origin='*', headers='Content-Type')
def content_post():
    try:
        ffile = request.files['file']
        bio = io.BytesIO()
        ffile.save(bio)
        fname = os.path.basename(ffile.filename)
        image = cv.imdecode(np.frombuffer(bio.getvalue(), dtype='uint8'), 1)
        
        if os.path.exists("temp") :
            shutil.rmtree("temp")
        os.mkdir("temp")
        cv2.imwrite("temp/temp.jpg",image)
        formated_output=inference_endpoint.inference("temp/temp.jpg", args['visualize'])
        
        
        status_code=200
        response = {
            'status': 'OK',
            'data':formated_output
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
