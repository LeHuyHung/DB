CUDA_VISIBLE_DEVICES=0 python train.py experiments/seg_detector/totaltext_resnet50_deform_thre.yaml --resume /home/asilla/10_dataset/hunglh/text_detection/DB_ajp10/DB/experiments/checkpoint/pre-trained-model-synthtext-resnet50 --num_gpus 1 --epochs=3000000 --batch_size=6 --start_epoch=0 --start_iter=198000
