import numpy as np

from concern import Logger, AverageMeter
from concern.config import Configurable
from concern.icdar2015_eval.detection.iou import DetectionIoUEvaluator


class QuadMeasurer(Configurable):
    def __init__(self, **kwargs):
        self.evaluator = DetectionIoUEvaluator()

    def measure(self, batch, output, is_output_polygon=False, box_thresh=0.6):
        '''
        batch: (image, polygons, ignore_tags
        batch: a dict produced by dataloaders.
            image: tensor of shape (N, C, H, W).
            polygons: tensor of shape (N, K, 4, 2), the polygons of objective regions.
            ignore_tags: tensor of shape (N, K), indicates whether a region is ignorable or not.
            shape: the original shape of images.
            filename: the original filenames of images.
        output: (polygons, ...)
        '''
        results = []
        gt_polyons_batch = batch['polygons']
        ignore_tags_batch = batch['ignore_tags']
        pred_polygons_batch = np.array(output[0])
        pred_scores_batch = np.array(output[1])
        for polygons, pred_polygons, pred_scores, ignore_tags in\
                zip(gt_polyons_batch, pred_polygons_batch, pred_scores_batch, ignore_tags_batch):
            gt = [dict(points=polygons[i], ignore=ignore_tags[i])
                  for i in range(len(polygons))]
            if is_output_polygon:
                pred = [dict(points=pred_polygons[i])
                        for i in range(len(pred_polygons))]
            else:
                pred = []
                # print(pred_polygons.shape)
                for i in range(pred_polygons.shape[0]):
                    if pred_scores[i] >= box_thresh:
                        # print(pred_polygons[i,:,:].tolist())
                        pred.append(dict(points=pred_polygons[i,:,:].tolist()))
                # pred = [dict(points=pred_polygons[i,:,:].tolist()) if pred_scores[i] >= box_thresh for i in range(pred_polygons.shape[0])]
            # print("============gt==================")
            # print(max([np.max(np.array(g['points']))for g in gt]))
            # print("===========pred===================")
            # print(max([np.max(np.array(g['points']))for g in pred]))
            # print(self.evaluator.evaluate_image)
            # print(len(gt), len(pred))
            # print(gt[0])
            # print(pred[0])
            pred_img=np.zeros((5000,5000),dtype=np.uint8)
            gt_img=np.zeros((5000,5000),dtype=np.uint8)
            import cv2
            for box in pred:
                cv2.drawContours(pred_img, [np.array(box['points'],dtype=np.int32)], 0, 255, 2)
            for box in gt:
                cv2.drawContours(gt_img, [np.array(box['points'],dtype=np.int32)], 0, 255, 2)
            cv2.imwrite("pred_img.jpg",pred_img)
            cv2.imwrite("gt_img.jpg",gt_img)
            1/0
            results.append(self.evaluator.evaluate_image(gt, pred))
            print(results[-1])
        return results

    def validate_measure(self, batch, output, is_output_polygon=False, box_thresh=0.6):
        return self.measure(batch, output, is_output_polygon, box_thresh)

    def evaluate_measure(self, batch, output):
        return self.measure(batch, output),\
            np.linspace(0, batch['image'].shape[0]).tolist()

    def gather_measure(self, raw_metrics, logger: Logger):
        raw_metrics = [image_metrics
                       for batch_metrics in raw_metrics
                       for image_metrics in batch_metrics]

        result = self.evaluator.combine_results(raw_metrics)

        precision = AverageMeter()
        recall = AverageMeter()
        fmeasure = AverageMeter()

        precision.update(result['precision'], n=len(raw_metrics))
        recall.update(result['recall'], n=len(raw_metrics))
        fmeasure_score = 2 * precision.val * recall.val /\
            (precision.val + recall.val + 1e-8)
        fmeasure.update(fmeasure_score)

        return {
            'precision': precision,
            'recall': recall,
            'fmeasure': fmeasure
        }
