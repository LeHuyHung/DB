import numpy as np
import cv2
import os
from shapely.geometry import Polygon
import pyclipper

from concern.config import State
from .data_process import DataProcess


class MakeSegAnglenData(DataProcess):
    r'''
    Making float angle mask from detection data with ICDAR format.
    Typically following the process of class `MakeICDARData`.
    '''
    min_text_size = State(default=8)
    shrink_ratio = State(default=0.4)

    def __init__(self, **kwargs):
        self.load_all(**kwargs)

    def process(self, data):
        '''
        requied keys:
            image, polygons, ignore_tags, filename
        adding keys:
            mask
        '''
        image = data['image']
        polygons = data['polygons']
        ignore_tags = data['ignore_tags']
        image = data['image']
        filename = data['filename']

        h, w = image.shape[:2]
        if data['is_training']:
            polygons, ignore_tags = self.validate_polygons(
                polygons, ignore_tags, h, w)
        angle_gt = np.zeros((1, h, w), dtype=np.float32)
        angle_mask = np.zeros((h, w), dtype=np.float32)
        for i in range(len(polygons)):
            polygon = polygons[i]
            height = max(polygon[:, 1]) - min(polygon[:, 1])
            width = max(polygon[:, 0]) - min(polygon[:, 0])
            # height = min(np.linalg.norm(polygon[0] - polygon[3]),
            #              np.linalg.norm(polygon[1] - polygon[2]))
            # width = min(np.linalg.norm(polygon[0] - polygon[1]),
            #             np.linalg.norm(polygon[2] - polygon[3]))
            if ignore_tags[i] or min(height, width) < self.min_text_size:
                cv2.fillPoly(angle_mask, polygon.astype(
                    np.int32)[np.newaxis, :, :], 0)
                ignore_tags[i] = True
            else:
                polygon_shape = Polygon(polygon)
                distance = polygon_shape.area * \
                    (1 - np.power(self.shrink_ratio, 2)) / polygon_shape.length
                subject = [tuple(l) for l in polygons[i]]
                padding = pyclipper.PyclipperOffset()
                padding.AddPath(subject, pyclipper.JT_ROUND,
                                pyclipper.ET_CLOSEDPOLYGON)
                shrinked = padding.Execute(-distance)
                if shrinked == []:
                    cv2.fillPoly(angle_mask, polygon.astype(
                        np.int32)[np.newaxis, :, :], 0)
                    ignore_tags[i] = True
                    continue
                shrinked = np.array(shrinked[0]).reshape(-1, 2)
                print(polygon)
                cv2.fillPoly(angle_gt[0], [shrinked.astype(np.int32)], 1)
                cv2.fillPoly(angle_mask, polygon.astype(
                    np.int32)[np.newaxis, :, :], 1)
        if filename is None:
            filename = ''
        data.update(angle_gt=angle_gt, angle_mask=angle_mask)
       
        dump_mask=np.array(np.stack([data['mask'][:,:]*254],axis=2),dtype=np.uint8)
        
        cv2.imwrite("dump_mask/"+str(len(os.listdir("dump_mask")))+".jpg",dump_mask)
        
        dump_angle_gt=np.array(np.stack([data['angle_gt'][0,:,:]*254],axis=2),dtype=np.uint8)
        
        cv2.imwrite("dump_angle_gt/"+str(len(os.listdir("dump_angle_gt")))+".jpg",dump_angle_gt)
        return data

    def validate_polygons(self, polygons, ignore_tags, h, w):
        '''
        polygons (numpy.array, required): of shape (num_instances, num_points, 2)
        '''
        if len(polygons) == 0:
            return polygons, ignore_tags
        assert len(polygons) == len(ignore_tags)
        for polygon in polygons:
            polygon[:, 0] = np.clip(polygon[:, 0], 0, w - 1)
            polygon[:, 1] = np.clip(polygon[:, 1], 0, h - 1)

        for i in range(len(polygons)):
            area = self.polygon_area(polygons[i])
            if abs(area) < 1:
                ignore_tags[i] = True
            if area > 0:
                polygons[i] = polygons[i][::-1, :]
        return polygons, ignore_tags

    def polygon_area(self, polygon):
        edge = 0
        for i in range(polygon.shape[0]):
            next_index = (i + 1) % polygon.shape[0]
            edge += (polygon[next_index, 0] - polygon[i, 0]) * (polygon[next_index, 1] - polygon[i, 1])

        return edge / 2.

