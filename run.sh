python classify.py \
    --dataset_file WHEE/processed_datasets/hri_data_cues_revised_label.csv \
    --agent_file agents/openai.yaml \
    --inputs_cols text \
    --save_dir results \
    --n 5 \
    --use_structured_outputs