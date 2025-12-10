for f in 0 1 2 3 4; do
    python train.py -cfg configs/mby140/mby140_train_pretrained.yaml --log_time fold4 --fold 1 --output_root output/not_val/bce/v1 --log_root log/not_val/bce/v1
done