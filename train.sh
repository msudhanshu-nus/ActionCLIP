for f in 0 1 2 3 4; do
    python train.py -cfg configs/mby140/mby140_train_pretrained.yaml --log_time fold4 --fold 4 --output_root pretrained_mby140 --log_root pretrained_mby140_logs
done