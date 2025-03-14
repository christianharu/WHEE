python classify.py \
    --dataset_file processed_datasets/hri_data_cues_revised_label.csv \
    --agent_file agents/openai.yaml \
    --inputs_cols text \
    --save_dir results \
    --n 20

python metrics.py \
    --dataset_file results/hri_data_cues_revised_label/gpt-4o-mini.csv \
    --label_col label \
    --prediction_col classification_label