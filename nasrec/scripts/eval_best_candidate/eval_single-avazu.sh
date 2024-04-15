LR=0.12
ID=0
score=1zico_1fr

CUDA_VISIBLE_DEVICES=7 python -u nasrec/main_train.py --root_dir ./data/avazu_kaggle_autoctr/ \
        --net supernet-config \
        --supernet_config ea-avazu-kaggle-autoctr-best-1shot-${score}/best_config_${ID}.json \
        --num_epochs 1 \
        --learning_rate ${LR} \
        --train_batch_size 512 \
        --wd 0 \
        --logging_dir ./www-test/ea-best-avazu-kaggle-${score}-config-${ID} \
        --gpu 0 \
        --test_interval 5000 \
        --dataset avazu
