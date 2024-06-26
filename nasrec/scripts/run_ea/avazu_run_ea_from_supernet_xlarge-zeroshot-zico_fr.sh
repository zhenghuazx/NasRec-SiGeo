zico_coef=0
fr_coef=1
loss_coef=1

CKPT_PATH=./www-test/avazu_1shot/avazu-supernet-default-binomial_0.5-xlarge_lr0.12/supernet_7blocks_layernorm1_default-binomial-0.5_lr0.12_supernetwarmup_12500/supernet_checkpoint.pt
LOGGING_DIR=./www-test/avazu_1shot/avazu-supernet-default-binomial-0.5-xlarge-ea-240gen-128pop-64sample-8childs-default-ft_lr0.04-${zico_coef}zico_${fr_coef}fr

python -u nasrec/eval_subnet_from_supernet.py \
    --root_dir ./data/avazu_kaggle_autoctr/ \
    --ea_top_k 2 \
    --ckpt_path $CKPT_PATH \
    --learning_rate 0.04 \
    --wd 0 \
    --logging_dir $LOGGING_DIR \
    --n_childs 8 \
    --n_generations 240 \
    --init_population 128 \
    --sample_size 64 \
    --method regularized-ea \
    --use_layernorm 1 \
    --max_train_steps 500 \
    --train_batch_size 512 \
    --test_batch_size 8192 \
    --num_parallel_workers 8 \
    --max_eval_steps 125 \
    --config xlarge \
    --test_only_at_last_step 1 \
    --finetune_whole_supernet 0 \
    --train_limit 32343176 \
    --test_limit 4042896  \
    --dataset avazu \
    --zeroshot zico-fr \
    --zico_coef ${zico_coef} \
   --fr_coef ${fr_coef} \
    --loss_coef ${loss_coef}
