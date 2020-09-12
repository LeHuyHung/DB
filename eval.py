#!python3
import argparse
import os
os.environ["CUDA_VISIBLE_DEVICES"]="2"
import torch
import yaml
from tqdm import tqdm
import numpy as np
from trainer import Trainer
import math
# tagged yaml objects
from experiment import Structure, TrainSettings, ValidationSettings, Experiment
from concern.log import Logger
from data.data_loader import DataLoader
from data.image_dataset import ImageDataset
from training.checkpoint import Checkpoint
from training.learning_rate import (
    ConstantLearningRate, PriorityLearningRate, FileMonitorLearningRate
)
from training.model_saver import ModelSaver
from training.optimizer_scheduler import OptimizerScheduler
from concern.config import Configurable, Config
import time
import cv2
import cv2 as cv
def main():
    parser = argparse.ArgumentParser(description='Text Recognition Training')
    parser.add_argument('exp', type=str)
    parser.add_argument('--batch_size', type=int,
                        help='Batch size for training')
    
    parser.add_argument('--resume', type=str, help='Resume from checkpoint')
    parser.add_argument('--result_dir', type=str, default='./results/', help='path to save results')
    parser.add_argument('--epochs', type=int, help='Number of training epochs')
    parser.add_argument('--start_iter', type=int,
                        help='Begin counting iterations starting from this value (should be used with resume)')
    parser.add_argument('--start_epoch', type=int,
                        help='Begin counting epoch starting from this value (should be used with resume)')
    parser.add_argument('--max_size', type=int, help='max length of label')
    parser.add_argument('--data', type=str,
                        help='The name of dataloader which will be evaluated on.')
    parser.add_argument('--thresh', type=float,
                        help='The threshold to replace it in the representers')
    parser.add_argument('--box_thresh', type=float, default=0.6,
                        help='The threshold to replace it in the representers')
    parser.add_argument('--verbose', action='store_true',
                        help='show verbose info')
    parser.add_argument('--no-verbose', action='store_true',
                        help='show verbose info')
    parser.add_argument('--visualize', action='store_true',
                        help='visualize maps in tensorboard')
    parser.add_argument('--resize', action='store_true',
                        help='resize')
    parser.add_argument('--polygon', action='store_true',
                        help='output polygons if true')
    parser.add_argument('--eager', '--eager_show', action='store_true', dest='eager_show',
                        help='Show iamges eagerly')
    parser.add_argument('--speed', action='store_true', dest='test_speed',
                        help='Test speed only')
    parser.add_argument('--dest', type=str,
                        help='Specify which prediction will be used for decoding.')
    parser.add_argument('--debug', action='store_true', dest='debug',
                        help='Run with debug mode, which hacks dataset num_samples to toy number')
    parser.add_argument('--no-debug', action='store_false',
                        dest='debug', help='Run without debug mode')
    parser.add_argument('-d', '--distributed', action='store_true',
                        dest='distributed', help='Use distributed training')
    parser.add_argument('--local_rank', dest='local_rank', default=0,
                        type=int, help='Use distributed training')
    parser.add_argument('-g', '--num_gpus', dest='num_gpus', default=1,
                        type=int, help='The number of accessible gpus')
    parser.add_argument('--image_short_side', type=int, default=736,
                        help='The threshold to replace it in the representers')
    parser.set_defaults(debug=False, verbose=False)

    args = parser.parse_args()
    args = vars(args)
    args = {k: v for k, v in args.items() if v is not None}

    conf = Config()
    experiment_args = conf.compile(conf.load(args['exp']))['Experiment']
    experiment_args.update(cmd=args)
    experiment = Configurable.construct_class_from_config(experiment_args)

    Eval(experiment, experiment_args, cmd=args, verbose=args['verbose']).eval(args['visualize'])

def skeletonlize(binarymap):
    img=binarymap
    # Step 1: Create an empty skeleton
    size = np.size(img)
    skel = np.zeros(img.shape, np.uint8)

    # Get a Cross Shaped Kernel
    element = cv2.getStructuringElement(cv2.MORPH_CROSS, (3,3))

    # Repeat steps 2-4
    while True:
        #Step 2: Open the image
        open = cv2.morphologyEx(img, cv2.MORPH_OPEN, element)
        #Step 3: Substract open from the original image
        temp = cv2.subtract(img, open)
        #Step 4: Erode the original image and refine the skeleton
        eroded = cv2.erode(img, element)
        skel = cv2.bitwise_or(skel,temp)
        img = eroded.copy()
        # Step 5: If there are no white pixels left ie.. the image has been completely eroded, quit the loop
        if cv2.countNonZero(img)==0:
            break
    return skel
    # Displaying the final skeleto
class Eval:
    def __init__(self, experiment, args, cmd=dict(), verbose=False):
        self.RGB_MEAN = np.array([122.67891434, 116.66876762, 104.00698793])
        self.experiment = experiment
        experiment.load('evaluation', **args)
        self.data_loaders = experiment.evaluation.data_loaders
        self.args = cmd
        self.logger = experiment.logger
        model_saver = experiment.train.model_saver
        self.structure = experiment.structure
        self.model_path = cmd.get(
            'resume', os.path.join(
                self.logger.save_dir(model_saver.dir_path),
                'final'))
        self.verbose = verbose

    def init_torch_tensor(self):
        # Use gpu or not
        torch.set_default_tensor_type('torch.FloatTensor')
        if torch.cuda.is_available():
           
            self.device = torch.device('cuda')
            torch.set_default_tensor_type('torch.cuda.FloatTensor')
        else:
            print(" cpu use")
            1/0
            self.device = torch.device('cpu')

    def init_model(self):
        model = self.structure.builder.build(self.device)
        return model

    def resume(self, model, path):
        if not os.path.exists(path):
            self.logger.warning("Checkpoint not found: " + path)
            return
        self.logger.info("Resuming from " + path)
        states = torch.load(
            path, map_location=self.device)
        model.load_state_dict(states, strict=False)
        self.logger.info("Resumed from " + path)

    def report_speed(self, model, batch, times=100):
        data = {k: v[0:1]for k, v in batch.items()}
        if  torch.cuda.is_available():
            torch.cuda.synchronize()
        start = time.time() 
        for _ in range(times):
            pred = model.forward(data)
        for _ in range(times):
            output = self.structure.representer.represent(batch, pred, is_output_polygon=False) 
        time_cost = (time.time() - start) / times
        self.logger.info('Params: %s, Inference speed: %fms, FPS: %f' % (
            str(sum(p.numel() for p in model.parameters() if p.requires_grad)),
            time_cost * 1000, 1 / time_cost))
        
        return time_cost
        
    def format_output(self, batch, output):
        batch_boxes, batch_scores = output
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
                        box = np.array(box).reshape(-1).tolist()
                        result = ",".join([str(int(x)) for x in box])
                        score = scores[i]
                        res.write(result + ',' + str(score) + "\n")
            else:
                with open(result_file_path, 'wt') as res:
                    for i in range(boxes.shape[0]):
                        score = scores[i]
                        if score < self.args['box_thresh']:
                            continue
                        box = boxes[i,:,:].reshape(-1).tolist()
                        result = ",".join([str(int(x)) for x in box])
                        res.write(result + ',' + str(score) + "\n")
    def load_image(self, image_path):
        img = cv2.imread(image_path, cv2.IMREAD_COLOR).astype('float32')
        original_shape = img.shape[:2]
        img = self.resize_image(img)
        img -= self.RGB_MEAN
        img /= 255.
        img = torch.from_numpy(img).permute(2, 0, 1).float().unsqueeze(0).to(self.device)
        return img, original_shape
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
    def eval(self, visualize=False):
        self.init_torch_tensor()
        model = self.init_model()
        self.resume(model, self.model_path)
        all_matircs = {}
        model.eval()
        vis_images = dict()
        with torch.no_grad():
            for _, data_loader in self.data_loaders.items():
                raw_metrics = []
                for i, batch in tqdm(enumerate(data_loader), total=len(data_loader)):
                    
                    if self.args['test_speed']:
                        time_cost = self.report_speed(model, batch, times=50)
                        continue
                    
                    
                    
                    #============= demo processs=========
                    if False:
                        image_path="/home/asilla/10_dataset/hunglh/text_detection/DB_notskip_bottom_line/DB/datasets/total_text/test_images/"+batch['filename'][0]
                        batch['filename'] = [image_path]
                        img, original_shape = self.load_image(image_path)
                        batch['shape'] = [original_shape]
                        with torch.no_grad():
                            batch['image'] = img
                            pred = model.forward(batch, training=False)
                            print(type(pred))
                            output = self.structure.representer.represent(batch, pred, is_output_polygon=self.args['polygon']) 
                            if not os.path.isdir(self.args['result_dir']):
                                os.mkdir(self.args['result_dir'])
                            self.format_output(batch, output)
                        raw_metric = self.structure.measurer.validate_measure(batch, output, is_output_polygon=self.args['polygon'], box_thresh=self.args['box_thresh'])
                        # print(type(raw_metric),len(raw_metric),type(raw_metric[0]),raw_metric[0].keys())
                        # print(raw_metric[0]['precision'],raw_metric[0]['hmean'],raw_metric[0]['recall'])
                        # input()
                        raw_metrics.append(raw_metric)
                        if True and self.structure.visualizer:
                            vis_image = self.structure.visualizer.visualize(batch, output, pred)
                            self.logger.save_image_dict(vis_image)
                            vis_images.update(vis_image)
                            print(type(vis_image))
                            print(vis_image.keys())
                            print('demo_results/'+batch['filename'][0].split(".")[0]+"_eval.jpg")
                            cv2.imwrite('demo_results/'+os.path.basename(batch['filename'][0]).split(".")[0]+"_eval.jpg",vis_image[batch['filename'][0]+"_output"])
                    #============= newprocess ====
                    elif True:
                        print(batch.keys())
                        image_path="/home/asilla/10_dataset/hunglh/text_detection/DB_notskip_bottom_line/DB/datasets/total_text/test_images/"+batch['filename'][0]
                        img, original_shape = self.load_image(image_path)
                        batch['shape'] = [original_shape]
                        batch['image'] = img
                        print("self.device",self.device)
                        pred_result = model.forward(batch, training=False)
                        pred=pred_result['thresh_binary']
                        pred_border=pred_result['thresh_binary_border']
                        border_dilation = cv2.dilate((pred_border.cpu().numpy()[0,0,:,:]>0.2).astype(np.uint8)*255,np.ones((5,5),np.uint8),iterations = 1)
                        border_dilation=skeletonlize(border_dilation)
                        border_dilation=cv2.dilate(border_dilation,np.ones((1,11),np.uint8),iterations = 1)
                        
                        dilation = cv2.dilate((pred.cpu().numpy()[0,0,:,:]>0.5).astype(np.uint8)*255,np.ones((5,5),np.uint8),iterations = 1)
                        
                        dilation = cv.bitwise_and(dilation,cv.bitwise_not(border_dilation))
                        cv2.imwrite('demo_results/'+batch['filename'][0].split(".")[0]+"_eval_dilation.jpg",dilation)
                        cv2.imwrite('demo_results/'+batch['filename'][0].split(".")[0]+"_eval_border.jpg",border_dilation)
                        dilation=np.array([[dilation]])
                        dilation=dilation/255
                        output = self.structure.representer.represent(batch, dilation, is_output_polygon=self.args['polygon']) 
                        dilation_border = (pred_border.cpu().numpy()[0,0,:,:]>0.001).astype(np.uint8)*255
                        self.format_output(batch, output)
                        raw_metric = self.structure.measurer.validate_measure(batch, output, is_output_polygon=self.args['polygon'], box_thresh=self.args['box_thresh'])
                        # print(type(raw_metric),len(raw_metric),type(raw_metric[0]),raw_metric[0].keys())
                        # print(raw_metric[0]['precision'],raw_metric[0]['hmean'],raw_metric[0]['recall'])
                        # input()
                        raw_metrics.append(raw_metric)
                        if True and self.structure.visualizer:
                            vis_image = self.structure.visualizer.visualize(batch, output, pred)
                            self.logger.save_image_dict(vis_image)
                            vis_images.update(vis_image)
                            print(type(vis_image))
                            print(vis_image.keys())
                            print('demo_results/'+batch['filename'][0].split(".")[0]+"_eval.jpg")
                            cv2.imwrite('demo_results/'+os.path.basename(batch['filename'][0]).split(".")[0]+"_eval.jpg",vis_image[batch['filename'][0]+"_output"])
                    #============= newprocess ====
                    else:
                        pred = model.forward(batch, training=False)
                        # print("=====is_output_polygon===",self.args['polygon'],type(batch))
                        # print(batch.keys())
                        # print(batch['filename'])
                        # print(batch['image'].shape)
                        # print(batch['gt'][0].shape)
                        output = self.structure.representer.represent(batch, pred, is_output_polygon=self.args['polygon']) 
                        if not os.path.isdir(self.args['result_dir']):
                            os.mkdir(self.args['result_dir'])
                        self.format_output(batch, output)
                        
                        raw_metric = self.structure.measurer.validate_measure(batch, output, is_output_polygon=self.args['polygon'], box_thresh=self.args['box_thresh'])
                        # print(type(raw_metric),len(raw_metric),type(raw_metric[0]),raw_metric[0].keys())
                        # print(raw_metric[0]['precision'],raw_metric[0]['hmean'],raw_metric[0]['recall'])
                        # input()
                        raw_metrics.append(raw_metric)

                        if visualize and self.structure.visualizer:
                            vis_image = self.structure.visualizer.visualize(batch, output, pred)
                            self.logger.save_image_dict(vis_image)
                            vis_images.update(vis_image)
                metrics = self.structure.measurer.gather_measure(raw_metrics, self.logger)
                for key, metric in metrics.items():
                    self.logger.info('%s : %f (%d)' % (key, metric.avg, metric.count))

if __name__ == '__main__':
    main()
