for f in 0 1 2 3 4; do
    python train.py -cfg configs/mby140/mby140_train_pretrained.yaml --log_time fold --fold 1 --loss_type bce --output_root output/plovad/bce --log_root log/plovad/bce
done