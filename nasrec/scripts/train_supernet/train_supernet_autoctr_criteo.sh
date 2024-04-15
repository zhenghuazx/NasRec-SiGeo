LR=0.12
WD=0

python -u nasrec/train_supernet.py \
  --learning_rate $LR \
  --gpu 0 \
  --root_dir ./data/criteo_kaggle_autoctr \
  --train_batch_size 512 \
  --strategy default \
  --anypath_choice binomial-0.5 \
  --logging_dir ./www-test/criteo_1shot/criteo-supernet-default-binomial_0.5-autoctr_lr${LR} \
  --test_batch_size 4096 \
  --use_layernorm 1 \
  --supernet_training_step 15000 \
  --config autoctr \
  --num_blocks 7 \
  --num_epochs 1 \
  --test_interval 2000 \
  --wd $WD \
  --train_limit 36672495 \
  --test_limit 4584061
