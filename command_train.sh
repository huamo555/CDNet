CUDA_VISIBLE_DEVICES=3 python train.py --camera realsense --log_dir "/data2/gaoyuming/mutiview_graspness/graspness_depthguji/output/train_nu1_re_1007" --batch_size 1 --learning_rate 0.001 --model_name minkuresunet --dataset_root "/data2/gaoyuming/.cache/datasets/dataset-data/" --resume --checkpoint_path "/data2/gaoyuming/mutiview_graspness/graspness_depthguji/output/train_nu1_re_1007/minkuresunet_epoch97.tar"

#"--camera","kinect",
#                "--dataset_root","/data2/gaoyuming/.cache/datasets/dataset-data/",
#                "--model_name", "minkuresunet"