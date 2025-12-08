for f in 0 1 2 3 4; do
    python train.py -cfg configs/mby140/mby140_train_pretrained.yaml --log_time fold${f} --fold ${f} --loss_type asl --output_root output/pretrained_mby140/asl1 --log_root log/pretrained_mby140_logs/asl1
done