DATASET_DIR=processed_datasets
RESULTS_DIR=train_results
MODEL=bert-large-uncased


# non_hri_data
CUDA_VISIBLE_DEVICES=5 python finetune_classify.py \
    --model_dir ${RESULTS_DIR}/${MODEL} \
    --dataset_file ${DATASET_DIR}/test/non_hri_data_test.csv \
    --inputs_col text \
    --save_file {model}.csv

python metrics.py \
    --dataset_file results/non_hri_data_test/${MODEL}.csv \
    --label_col label \
    --prediction_col classification_label


# hri_data
CUDA_VISIBLE_DEVICES=5 python finetune_classify.py \
    --model_dir ${RESULTS_DIR}/${MODEL} \
    --dataset_file ${DATASET_DIR}/hri_data_cues_revised_label.csv \
    --inputs_col text \
    --save_file {model}.csv

python metrics.py \
    --dataset_file results/hri_data_cues_revised_label/${MODEL}.csv \
    --label_col revised_label \
    --prediction_col classification_label
