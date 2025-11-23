for f in 0 1 2 3 4; do
    python test.py -cfg configs/mby140/mby140_test.yaml --log_time fold${f}_test --fold ${f} --split test
done