#CUDA_VISIBLE_DEVICES=2 python eval.py experiments/seg_detector/totaltext_resnet50_deform_thre.yaml --resume /home/asilla/10_dataset/hunglh/text_detection/DB/outputs/workspace/DB/SegDetectorModel-seg_detector/deformable_resnet50_version1/L1BalanceCELoss/model/model_epoch_1192_minibatch_198000 --polygon --box_thresh 0.6 --batch_size=1

CUDA_VISIBLE_DEVICES=0 python eval.py experiments/seg_detector/totaltext_resnet50_deform_thre.yaml --resume /home/asilla/10_dataset/hunglh/text_detection/DB_notskip_bottom_line/DB/workspace/SegDetectorModel-seg_detector/deformable_resnet50/L1BalanceCELoss/model/model_epoch_2864_minibatch_1908000 --batch_size=1 --box_thresh 0.3 #--box_thresh 0.3 


