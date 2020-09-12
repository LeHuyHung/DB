#CUDA_VISIBLE_DEVICES=0 python demo.py experiments/seg_detector/totaltext_resnet50_deform_thre.yaml --image_path /home/asilla/10_dataset/hunglh/text_detection/data/thaysang_pdf/images/ --resume /home/asilla/10_dataset/hunglh/text_detection/DB/outputs/workspace/DB/SegDetectorModel-seg_detector/deformable_resnet50/L1BalanceCELoss/model/model_epoch_2927_minibatch_684000 --polygon --box_thresh 0.02

#CUDA_VISIBLE_DEVICES=0 python demo.py experiments/seg_detector/totaltext_resnet50_deform_thre.yaml --image_path /home/asilla/10_dataset/hunglh/text_detection/data/east_detect/final_testset_allRect/images --resume /home/asilla/10_dataset/hunglh/text_detection/DB/outputs/workspace/DB/SegDetectorModel-seg_detector/deformable_resnet50/L1BalanceCELoss/model/model_epoch_2927_minibatch_684000 --polygon --box_thresh 0.02

#CUDA_VISIBLE_DEVICES=0 python demo.py experiments/seg_detector/totaltext_resnet50_deform_thre.yaml --image_path /home/asilla/10_dataset/hunglh/text_detection/data/debug --resume /home/asilla/10_dataset/hunglh/text_detection/DB/outputs/workspace/DB/SegDetectorModel-seg_detector/deformable_resnet50/L1BalanceCELoss/model/model_epoch_2927_minibatch_684000 --polygon --box_thresh 0.02




CUDA_VISIBLE_DEVICES=0 python demo.py experiments/seg_detector/totaltext_resnet50_deform_thre.yaml --image_path /home/asilla/10_dataset/hunglh/text_detection/data/receipt --resume /home/asilla/10_dataset/hunglh/text_detection/DB/outputs/workspace/DB/SegDetectorModel-seg_detector/deformable_resnet50/L1BalanceCELoss/model/model_epoch_2927_minibatch_684000 --polygon --box_thresh 0.02
