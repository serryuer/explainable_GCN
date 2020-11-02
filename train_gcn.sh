python train_gcn.py \
    -lr 0.01 \
    -dataset_name mr \
    -dropout 0.5 \
    -epochs 100 \
    -gcn_layers 2 \
    -embed_dim 200 \
    -embed_fintune True \
    -hidden_size 256 \
    -node_size 29426 \
    -random_seed 42 \
    -device 2 \
    -model_name mr_id \
    -save_best_model False \
    -output_size 2 > 0.01_256_200_layer_1.log 2>&1 

python train_gcn.py \
    -lr 0.01 \
    -dataset_name mr \
    -dropout 0.5 \
    -epochs 200 \
    -gcn_layers 2 \
    -embed_dim 200 \
    -embed_fintune True \
    -hidden_size 256 \
    -node_size 29426 \
    -random_seed 42 \
    -device 2 \
    -model_name mr_id \
    -save_best_model True \
    -output_size 2 > 0.01_256_200_layer_2.log 2>&1 

python train_gcn.py \
    -lr 0.01 \
    -dataset_name mr \
    -dropout 0.5 \
    -epochs 100 \
    -gcn_layers 3 \
    -embed_dim 200 \
    -embed_fintune True \
    -hidden_size 256 \
    -node_size 29426 \
    -random_seed 42 \
    -device 2 \
    -model_name mr_id \
    -save_best_model True \
    -output_size 2 > 0.01_256_200_layer_3.log 2>&1 

python train_gcn.py \
    -lr 0.001 \
    -dataset_name mr \
    -dropout 0.5 \
    -epochs 100 \
    -gcn_layers 4 \
    -embed_dim 200 \
    -embed_fintune True \
    -hidden_size 256 \
    -node_size 29426 \
    -random_seed 42 \
    -device 2 \
    -model_name mr_id \
    -save_best_model False \
    -output_size 2 > 0.01_256_200_layer_4.log 2>&1 
