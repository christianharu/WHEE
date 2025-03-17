MODEL_NAME=google-bert/bert-base-uncased
DATASET_DIR=processed_datasets

CUDA_VISIBLE_DEVICES=4 python finetune.py \
    --model_name $MODEL_NAME \
    --dataset_files "{'train': '${DATASET_DIR}/train/non_hri_data_train.csv', 'validation': '${DATASET_DIR}/val/non_hri_data_val.csv', 'test': '${DATASET_DIR}/test/non_hri_data_test.csv'}" \
    --inputs_col text \
    --labels_col label \
    --output_dir train_results
