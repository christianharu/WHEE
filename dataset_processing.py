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


def get_VA(data):
    lexicon_df,wnl,stp_wrds = lexicon.setup_lexicon('cue_utilities/nrc_vad_lexicon/BipolarScale/NRC-VAD-Lexicon.txt')
    data['vad'] = data['text'].apply(lexicon.get_avg_vad, args = (lexicon_df,wnl,stp_wrds)) 
    data[['valence','arousal','dominance']] = pd.DataFrame(data.vad.tolist(),index = data.index)   
    return data.drop(columns=['dominance','vad'])


def get_sentiment_label(dataframe_row,mdl,tokenzr,utt_column):
    #gets the sentiment in accordance to the label we want
    #0 - negative, 1 - neutral, 2 - positive
    label_val = ['negative','neutral', 'positive']
    sentiment_lst = sp.get_sentiment(str(dataframe_row[utt_column]),mdl,tokenzr)
    #print(sentiment_lst)
    index_max = max(range(len(sentiment_lst)), key=sentiment_lst.__getitem__)
    return label_val[index_max]

def get_sentiment(data):
    sent_model, sent_tokenzr = sp.loadSentimentModel() #get model and tokenizer
    data['sentiment'] = data.apply(get_sentiment_label,axis = 1, args = (sent_model,sent_tokenzr,'text'))
    return data


def main():

    #read all datasets
    dir_path = os.path.dirname(os.path.realpath(__file__))
    iempathize = pd.read_csv(dir_path + '/iempathize.csv',encoding='windows-1252' )
    tweetemp = pd.read_csv(dir_path + '/TwittEmp.csv', )

    eerobot = pd.read_csv(dir_path + '/EERobot.csv', )
    ex = pd.read_csv(dir_path + '/EmpatheticExchanges_extended_v2.0.csv')
    tsc = pd.read_csv(dir_path + '/TSC.csv')
    #key none = 0, seek = 1, provide = 2


    tweetemp = tweetemp.rename(columns={"content": "text"})
    joined_df = pd.concat([tweetemp,iempathize])

    tsc = tsc.rename(columns={"final_label": "empathy"})

    joined_df.reset_index()
    joined_df = joined_df.drop(columns=['id','permalink'])

    extracted_listener_ex = get_utterances(ex,'listener')
    extracted_speaker_ex = get_utterances(ex,'speaker')
    extracted_listener_eer = get_utterances(eerobot,'listener')
    extracted_speaker_eer = get_utterances(eerobot,'speaker')
    extracted_listener_tsc = get_utterances_no_seek(tsc,'listener')
    extracted_speaker_tsc = get_utterances_no_seek(tsc,'speaker')

    computer_df = pd.concat([joined_df,extracted_listener_ex, extracted_speaker_ex], ignore_index=True)
    robotic_df = pd.concat([extracted_listener_eer,extracted_speaker_eer,extracted_listener_tsc,extracted_speaker_tsc], ignore_index=True)


    computer_df['text'] = computer_df['text'].apply(change_commas) 
    robotic_df['text'] = robotic_df['text'].apply(change_commas) 

    
    computer_df.to_csv('./processed_datasets/non_hri_data.csv', index=False)
    robotic_df.to_csv('./processed_datasets/hri_data.csv', index=False)


    #get cues
    computer_df_cues = get_VA(computer_df)
    computer_df_cues = get_sentiment(computer_df_cues)
    computer_df_cues = epitome.predict_epitome_values('cue_utilities/epitome_mechanisms/trained_models',computer_df_cues)
    computer_df_cues.to_csv('./processed_datasets/non_hri_data_cues.csv', index=False)
    
    robotic_df_cues = get_VA(robotic_df)
    robotic_df_cues = get_sentiment(robotic_df_cues)
    robotic_df_cues = epitome.predict_epitome_values('cue_utilities/epitome_mechanisms/trained_models',robotic_df_cues)
    robotic_df_cues.to_csv('./processed_datasets/hri_data_cues.csv', index=False)


    print(computer_df_cues)

    current_system_pid = os.getpid()
    ThisSystem = psutil.Process(current_system_pid)
    ThisSystem.terminate()





if __name__ == "__main__":
    main()