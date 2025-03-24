AGENT=llama-3.3
MODEL=Meta-Llama-3.3-70B-Instruct-AWQ-INT4

python classify.py \
    --dataset_file EERobot.csv \
    --agent_file agents/${AGENT}.yaml \
    --inputs_cols "['speaker_utterance']" \
    --save_file {model}.csv

python generate.py \
    --dataset_file results/EERobot/${MODEL}.csv \
    --agent_file agents/haru.yaml \
    --inputs_cols "['speaker_utterance']"

python classify.py \
    --dataset_file results/EERobot/${MODEL}_generate.csv \
    --agent_file agents/${AGENT}.yaml \
    --inputs_cols "['llm_utterance']" \
    --save_file {model}_generate_classify.csv
