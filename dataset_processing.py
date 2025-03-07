import pandas as pd 
import os


def get_utterances(data, utt_type):
    processed_df = data.copy()
    if utt_type == 'listener':       
        processed_df["label"] = processed_df.apply(lambda x: 0 if x["empathy"] == 1 else 2,axis=1)
        processed_df = processed_df.rename(columns={"listener_utterance": "text"})

        processed_df = processed_df.drop(columns = processed_df.columns.difference(['text','label']))
        return processed_df
    else:
        processed_df = processed_df.rename(columns={"speaker_utterance": "text"})
        processed_df['label'] = 1
        processed_df = processed_df.drop(columns = processed_df.columns.difference(['text','label']))
        return processed_df





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
    extracted_listener_tsc = get_utterances(tsc,'listener')
    extracted_speaker_tsc = get_utterances(tsc,'speaker')

    computer_df = pd.concat([joined_df,extracted_listener_ex, extracted_speaker_ex], ignore_index=True)
    robotic_df = pd.concat([extracted_listener_eer,extracted_speaker_eer,extracted_listener_tsc,extracted_speaker_tsc], ignore_index=True)
    
    computer_df.to_csv('./processed_datasets/non_hri_data.csv', index=False)
    robotic_df.to_csv('./processed_datasets/hri_data.csv', index=False)












if __name__ == "__main__":
    main()