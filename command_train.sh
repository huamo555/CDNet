CUDA_VISIBLE_DEVICES=0 python train.py --camera realsense --log_dir "/data3/gaoyuming/Graspgan_quan/depthguji_quan_huigui_time/output/train_nu1_re_0623_dfnet512" --batch_size 1 --learning_rate 0.001 --model_name minkuresunet --dataset_root "/data3/gaoyuming/project/datasets/datasets/dataset-data"
#"--camera","kinect",
#                "--dataset_root","/data2/gaoyuming/.cache/datasets/dataset-data/",
#                "--model_name", "minkuresunet"