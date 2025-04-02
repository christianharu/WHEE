AGENT=llama-3.1
MODEL=Meta-Llama-3.1-70B-Instruct-AWQ-INT4

# Default
python classify.py \
    --dataset_file processed_datasets/hri_data_cues_revised_label.csv \
    --agent_file agents/${AGENT}.yaml \
    --inputs_cols "['text']" \
    --save_file {model}.csv

python metrics.py \
    --dataset_file results/hri_data_cues_revised_label/${MODEL}.csv \
    --label_col revised_label \
    --prediction_col classification_label

# With cues
python classify_with_cues.py \
    --dataset_file processed_datasets/hri_data_cues_revised_label.csv \
    --agent_file agents/${AGENT}_with_cues.yaml \
    --inputs_cols "['text', 'arousal', 'valence', 'sentiment', 'predictions_ER', 'predictions_IP', 'predictions_EX']" \
    --save_file {model}_with_cues.csv

python metrics.py \
    --dataset_file results/hri_data_cues_revised_label/${MODEL}_with_cues.csv \
    --label_col revised_label \
    --prediction_col classification_label