DATASET_DIR=processed_datasets
RESULTS_DIR=train_results


# google-bert/bert-large-uncased
CUDA_VISIBLE_DEVICES=4 python finetune.py \
    --model_name google-bert/bert-large-uncased \
    --dataset_files "{'train': '${DATASET_DIR}/train/non_hri_data_train.csv', 'validation': '${DATASET_DIR}/val/non_hri_data_val.csv', 'test': '${DATASET_DIR}/test/non_hri_data_test.csv'}" \
    --inputs_col text \
    --labels_col label \
    --output_dir ${RESULTS_DIR}/bert-large-uncased


# FacebookAI/roberta-large
CUDA_VISIBLE_DEVICES=4 python finetune.py \
    --model_name FacebookAI/roberta-large \
    --dataset_files "{'train': '${DATASET_DIR}/train/non_hri_data_train.csv', 'validation': '${DATASET_DIR}/val/non_hri_data_val.csv', 'test': '${DATASET_DIR}/test/non_hri_data_test.csv'}" \
    --inputs_col text \
    --labels_col label \
    --output_dir ${RESULTS_DIR}/roberta-large


# answerdotai/ModernBERT-large
CUDA_VISIBLE_DEVICES=4 python finetune.py \
    --model_name answerdotai/ModernBERT-large \
    --dataset_files "{'train': '${DATASET_DIR}/train/non_hri_data_train.csv', 'validation': '${DATASET_DIR}/val/non_hri_data_val.csv', 'test': '${DATASET_DIR}/test/non_hri_data_test.csv'}" \
    --inputs_col text \
    --labels_col label \
    --output_dir ${RESULTS_DIR}/modernbert-large
