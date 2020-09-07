CUDA_VISIBLE_DEVICES=0,1 python train.py experiments/seg_detector/totaltext_resnet50_deform_thre.yaml --resume experiments/checkpoint/pre-trained-model-synthtext-resnet50 --num_gpus 2 --epochs=3000
