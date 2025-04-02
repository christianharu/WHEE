import pandas as pd 
import os
import re
from cue_utilities.nrc_vad_lexicon import lexicon_analysis as lexicon
from cue_utilities.sentiment import sentiment_prediction as sp
from cue_utilities.epitome_mechanisms import epitome_predictor as epitome

import psutil

def get_utterances(data, utt_type):
    processed_df = data.copy()
    if utt_type == 'listener':       
        processed_df["label"] = processed_df.apply(lambda x: 0 if x["empathy"] == 1 else 2,axis=1)
        processed_df = processed_df.rename(columns={"listener_utterance": "text"})

        processed_df = processed_df.drop(columns = processed_df.columns.difference(['text','label']))
        return processed_df
    else:
        processed_df['is_start'] = ~processed_df['id'].duplicated()
        processed_df = processed_df.rename(columns={"speaker_utterance": "text"})
        processed_df["label"] = processed_df.apply(lambda x: 1 if x["is_start"] == 1 else 0,axis=1)
        processed_df = processed_df.drop(columns = processed_df.columns.difference(['text','label']))
        return processed_df

def get_utterances_no_seek(data, utt_type):
    processed_df = data.copy()
    if utt_type == 'listener':       
        processed_df["label"] = processed_df.apply(lambda x: 0 if x["empathy"] == 1 else 2,axis=1)
        processed_df = processed_df.rename(columns={"listener_utterance": "text"})

        processed_df = processed_df.drop(columns = processed_df.columns.difference(['text','label']))
        return processed_df
    else:
        processed_df = processed_df.rename(columns={"speaker_utterance": "text"})
        processed_df['label'] = 0
        processed_df = processed_df.drop(columns = processed_df.columns.difference(['text','label']))
        return processed_df


def change_commas(utterance):
    utterance = str(utterance)
    utterance = re.sub("_comma_", ',', utterance)
    return utterance


def get_VA(data,target):
    lexicon_df,wnl,stp_wrds = lexicon.setup_lexicon('cue_utilities/nrc_vad_lexicon/BipolarScale/NRC-VAD-Lexicon.txt')
    data['vad'] = data[target].apply(lexicon.get_avg_vad, args = (lexicon_df,wnl,stp_wrds)) 
    data[[str(target)+'_valence',str(target)+'_arousal',str(target)+'_dominance']] = pd.DataFrame(data.vad.tolist(),index = data.index)   
    return data.drop(columns=[str(target)+'_dominance','vad'])


def get_sentiment_label(dataframe_row,mdl,tokenzr,utt_column):
    #gets the sentiment in accordance to the label we want
    #0 - negative, 1 - neutral, 2 - positive
    label_val = ['negative','neutral', 'positive']
    sentiment_lst = sp.get_sentiment(str(dataframe_row[utt_column]),mdl,tokenzr)
    #print(sentiment_lst)
    index_max = max(range(len(sentiment_lst)), key=sentiment_lst.__getitem__)
    return label_val[index_max]

def get_sentiment(data,target):
    sent_model, sent_tokenzr = sp.loadSentimentModel() #get model and tokenizer
    data[str(target)+'_sentiment'] = data.apply(get_sentiment_label,axis = 1, args = (sent_model,sent_tokenzr,str(target)))
    return data


def main():

    #read all datasets
    dir_path = os.path.dirname(os.path.realpath(__file__))

    eerobot = pd.read_csv(dir_path + '/results/EERobot/Meta-Llama-3.3-70B-Instruct-AWQ-INT4_generate_classify.csv', )
    tsc = pd.read_csv(dir_path + '/results/TSC/Meta-Llama-3.3-70B-Instruct-AWQ-INT4_generate_classify.csv')
    #key none = 0, seek = 1, provide = 2
    tsc = tsc.rename(columns={"final_label": "empathy"})
    print(eerobot.head())

    joined_df = pd.concat([eerobot,tsc])
    print(joined_df.head())
    joined_df.reset_index()
    
    joined_df['speaker_utterance'] = joined_df['speaker_utterance'].str.strip()
    joined_df['listener_utterance'] = joined_df['listener_utterance'].str.strip()
    joined_df['llm_utterance'] = joined_df['llm_utterance'].str.strip()

    print(joined_df.columns)

    joined_df = get_VA(joined_df,'speaker_utterance')
    joined_df = get_VA(joined_df,'listener_utterance')
    joined_df = get_VA(joined_df,'llm_utterance')

    print(joined_df.head())

    joined_df = get_sentiment(joined_df,'speaker_utterance')
    joined_df = get_sentiment(joined_df,'listener_utterance')
    joined_df = get_sentiment(joined_df,'llm_utterance')


    #joined_df = epitome.predict_epitome_values('cue_utilities/epitome_mechanisms/trained_models',joined_df,'listener_utterance')
    #joined_df = epitome.predict_epitome_values('cue_utilities/epitome_mechanisms/trained_models',joined_df,'llm_utterance')

    print(joined_df.columns)



    joined_df["empathetic"] = joined_df.apply(lambda x: 1 if len(str(x["instruction"])) > 3 else 0,axis=1)

    joined_df.to_csv('./results/hri_generated_responses_with_cues.csv', index=False)

    #joined_df.drop(columns=['listener_utterance_dominance','speaker_utterance_dominance']).to_csv('./results/hri_generated_responses_with_cues.csv', index=False)


    current_system_pid = os.getpid()
    ThisSystem = psutil.Process(current_system_pid)
    ThisSystem.terminate()


    #get cues
    #robotic_df_cues = get_VA(tsc,)
    #robotic_df.to_csv('./processed_datasets/hri_data.csv', index=False)
    #robotic_df_cues = get_sentiment(robotic_df_cues)
    #robotic_df.to_csv('./processed_datasets/hri_data.csv', index=False)
    #robotic_df_cues = epitome.predict_epitome_values('cue_utilities/epitome_mechanisms/trained_models',robotic_df_cues)
    #robotic_df_cues.to_csv('./processed_datasets/hri_data_cues.csv', index=False)
    #print(robotic_df_cues)




if __name__ == "__main__":
    main()