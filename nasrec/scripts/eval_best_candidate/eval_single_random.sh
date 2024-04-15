LR=0.1
ID=15
gup_id=7
data=avazu # criteo-kaggle
datapath=${data}_kaggle_autoctr
score=random

# CUDA_VISIBLE_DEVICES=7
python -u nasrec/main_train.py --root_dir ./data/${datapath}/ \
        --net supernet-config \
        --supernet_config ea-${data}-kaggle-autoctr-best-1shot-${score}/best_config_${ID}.json \
        --num_epochs 1 \
        --learning_rate ${LR} \
        --train_batch_size 256 \
        --wd 0 \
        --logging_dir ./www-test/ea-best-${data}-kaggle-${score}-config-${ID} \
        --gpu ${gup_id} \
        --test_interval 10000 \
        --dataset ${data} \
        --train_limit 36386071 \
        --test_limit 4042896
