#!/usr/bin/env python
# coding: utf-8
import wfdb
import ast
import numpy as np
import pandas as pd
import os
from tqdm import tqdm
import re
from google.cloud import translate_v3 as translate
import openai


PROJECT_ID = ""
assert PROJECT_ID
PARENT = f"projects/{PROJECT_ID}"

OPENAI_API_KEY = ""
openai.api_key = OPENAI_API_KEY


def load_raw_data(df, sampling_rate, path):
    if sampling_rate == 100:
        file_names = df.filename_lr
    else:
        file_names = df.filename_hr

    data = [wfdb.rdsamp(os.path.join(path, f)) for f in file_names]
    lead_signals = np.asarray([signal for signal, meta in data])
    lead_names = np.asarray([meta['sig_name'] for signal, meta in data])

    return lead_signals, lead_names


def main(**kwargs):
    # load and convert annotation data
    data_dir = kwargs["data_dir"]
    data_dir_processed = kwargs["data_processed_dir"]
    sampling_rate = kwargs["sampling_rate"]

    annotation_path = os.path.join(data_dir, 'ptbxl_database.csv')
    assert os.path.isfile(annotation_path), f"Can not find ptbxl_database.csv in {data_dir}.\n" \
                                            f"Please check directory of the PTB-XL data set in the config."
    annotation = pd.read_csv(annotation_path)
    annotation.scp_codes = annotation.scp_codes.apply(lambda x: ast.literal_eval(x))

    # load scp_statements.csv for diagnostic aggregation
    agg_df = pd.read_csv(os.path.join(data_dir, 'scp_statements.csv'), index_col=0)
    agg_df = agg_df[agg_df.diagnostic == 1]

    def aggregate_diagnostic(y_dic):
        temp = []
        for key in y_dic.keys():
            if key in agg_df.index:
                c = agg_df.loc[key].diagnostic_class
                if str(c) != 'nan':
                    temp.append(c)
                else:
                    print("Find nan.")
        return list(set(temp))

    # apply diagnostic superclass
    annotation['diagnostic_superclass'] = annotation.scp_codes.apply(aggregate_diagnostic)
    annotation['superdiagnostic_len'] = annotation['diagnostic_superclass'].apply(lambda x: len(x))

    # filter out cases with 0 diagnostic superclass
    annotation = annotation[annotation['superdiagnostic_len'] > 0]

    # count number of cases in each diagnostic superclass
    counts = pd.Series(np.concatenate(annotation.diagnostic_superclass.values)).value_counts()
    print(f"Case in each diagnostic superclass\n{counts}\n")

    # save lead data to csv files
    csv_dir = os.path.join(data_dir, "csv")
    os.makedirs(csv_dir, exist_ok=True)
    file_names = []
    labels = []
    diagnosis = []
    bar = tqdm(range(len(annotation.index)))

    for _, row in annotation.iterrows():
        bar.update()

        # get label of the case
        file_name = "PTB-XL-" + str(row["ecg_id"]).zfill(5)
        label = " ".join(row["diagnostic_superclass"])
        file_names.append(file_name)
        labels.append(label)

        # get diagnosis
        report = row['report']
        diagnosis.append(report)

        # load raw data
        if sampling_rate == 100:
            raw_file_name = row["filename_lr"]
        else:
            raw_file_name = row["filename_hr"]
        leads, meta = wfdb.rdsamp(os.path.join(data_dir, raw_file_name))
        leads = (1000 * leads).astype(int)
        lead_names = meta['sig_name']

        # convert raw data to dataframe
        data = {}
        for i, lead_name in enumerate(lead_names):
            if lead_name in ["AVR", "AVL", "AVF"]:
                lead_name = "a" + lead_name[1:]
            data[lead_name] = leads[:, i]
        # save to csv files
        csv_file = file_name + ".csv"
        csv_dir = os.path.join(data_dir_processed, "csv")
        csv_path = os.path.join(csv_dir, csv_file)
        os.makedirs(csv_dir, exist_ok=True)
        df = pd.DataFrame.from_dict(data)
        df.to_csv(csv_path, index=False)

    # save labels
    df = pd.DataFrame.from_dict(dict(file_name=file_names, label=labels, diagnosis=diagnosis))
    df.to_csv(os.path.join(data_dir_processed, "label.csv"), index=False)


def translate_text(text_list: [str], target_language_code: str) -> [translate.Translation]:
    client = translate.TranslationServiceClient()
    response = client.translate_text(
        parent=PARENT,
        contents=text_list,
        target_language_code=target_language_code,
    )
    return response.translations


def translate_diagnosis(**kwargs):
    data_dir_processed = kwargs["data_processed_dir"]
    df = pd.read_csv(os.path.join(data_dir_processed, "label.csv"))
    text_list = df["diagnosis"].to_list()
    text_list = [re.sub('  +', '. ', report) for report in text_list]

    text_list_unique = list(set(text_list))
    source_languages = {}
    translated_dict = {}

    batch_size = 150
    start_idx = 0
    length = len(text_list_unique)
    batch_num = length // batch_size + 1
    print("Translate message num:", length)

    bar = tqdm(range(batch_num))
    while start_idx < length:
        bar.update()
        batch_text = text_list_unique[start_idx: start_idx + batch_size]
        res = translate_text(batch_text, 'en')
        for i, r in enumerate(res):
            source_language = r.detected_language_code
            source_languages[source_language] = source_languages.get(source_language, 0) + 1
            translated_text = r.translated_text
            translated_dict[batch_text[i]] = translated_text
        start_idx += batch_size

    translated_list = [translated_dict[x] for x in text_list]

    df["diagnosis_en"] = translated_list
    df.to_csv(os.path.join(data_dir_processed, "label.csv"), index=False)

    texts = '\n'.join(translated_list)
    print("Source languages:", source_languages)
    with open("results.txt", 'w') as f:
        f.write(texts)


def correct_translation(x):
    # Set up the prompt for the conversation
    prompt = "Some symbols in the following ECG diagnosis is missing, " \
             "please added them and only return the diagnosis. " \
             f"{x}\n"

    # Set up the OpenAI API parameters
    model_engine = "text-davinci-002"
    temperature = 0.7
    max_tokens = 150

    # Start the conversation loop
    response = openai.Completion.create(
        engine=model_engine,
        prompt=prompt,
        temperature=temperature,
        max_tokens=max_tokens,
        n=1,
        stop=None,
        frequency_penalty=0,
        presence_penalty=0
    )

    # Print the AI response
    message = response.choices[0].text.strip()
    print("AI: " + message)


def simplify_diagnosis(**kwargs):
    data_dir_processed = kwargs["data_processed_dir"]
    df = pd.read_csv(os.path.join(data_dir_processed, "label.csv"))
    text_list = df["diagnosis_en"].to_list()
    simplified_text_list = []
    for text in text_list:
        text = text.lower()

        # remove edit and comparison results
        if " edit: " in text:
            text = text[:text.rindex(" edit: ")]
        if " compared with" in text:
            text = text[:text.rindex(" compared with")]

        # remove extra spaces
        text = text.strip(' ')
        text = re.sub(r' +', ' ', text)

        # remove in correct words and symbolises
        text = re.sub('ECG|ecg|EKG|ekg', 'ecg', text)
        text = re.sub(r'4\.46|unconfirmed report|unconfirmed be', '', text)
        text = re.sub('st &amp; t', 'st and t', text)
        for condition in [
            'sinus rhythm',
            'sinus bradycardia',
            'sinus arrhythmia',
            'atrial fibrillation/flutter',
            'atrial fibrillation/blocked',
            'atrial fibrillation',
            'atrial fibrillation/-normocardium',
            'accelerated av rhythm',
            'a rapid'
        ]:
            text = re.sub(f'^{condition} |^{condition}. ', f'{condition}, ', text)

        text = re.sub(r'type normal ', r'type normal. ', text)
        text = re.sub(r' incl$', '', text)
        text = re.sub(r' ?- ?', r'-', text)
        text = re.sub(r'left-type', r'left type', text)

        # remove extra period symbol atrial fibrillation. with
        text = re.sub(r'normocardes-\. tachycardic', r'normocardes-tachycardic', text)
        text = re.sub(r'atrial fibrillation\. with', r'atrial fibrillation with', text)
        text = re.sub(r'heart\. disease', r'heart disease', text)
        text = re.sub(r'abnormal, probable\. ', r'abnormal, probable ', text)
        text = re.sub(r'due to\. ', r'due to ', text)
        text = re.sub(r'due\. to', r'due to', text)
        text = re.sub(r'and\. ', r'and ', text)
        text = re.sub(r'inferolateral\. lead', r'inferolateral lead', text)
        text = re.sub(r'due to\. ', r'due to ', text)
        text = re.sub(r'as at\. ', r'as at ', text)

        # Upper certain words
        # for word in ['lahb']:
        #     text = re.sub(f'({word})', word.upper(), text)

        # remove incorrect space and symbolises
        text = text.strip(' ')
        text = re.sub(' +', ' ', text)
        text = re.sub(r'[ ,.]+\.', '.', text)
        if not text.endswith('.') and len(text) > 0:
            text = text + '.'

        simplified_text_list.append(text)

    df["diagnosis_en_simplified"] = simplified_text_list
    df.to_csv(os.path.join(data_dir_processed, "label.csv"), index=False)

    tmp = simplified_text_list
    texts = '\n'.join(tmp)
    with open("results.txt", 'w') as f:
        f.write(texts)

    text_list = df["diagnosis"].to_list()
    texts = '\n'.join(text_list)
    with open("original_results.txt", 'w') as f:
        f.write(texts)

    text_list = df["diagnosis_en"].to_list()
    texts = '\n'.join(text_list)
    with open("translated_results.txt", 'w') as f:
        f.write(texts)


if __name__ == '__main__':
    config = dict(
        data_dir="data/original/PTB-XL",
        data_processed_dir="data/decoded/PTB-XL",
        sampling_rate=100,
    )
    # main(**config)
    # translate_diagnosis(**config)
    simplify_diagnosis(**config)

    # Some symbols in the following ECG diagnosis is missing, please added them and only return the diagnosis.
    # sinus rhythm rs # transition to v leads shifted to the right incomplete right bundle branch block otherwise normal ekg
    # correct_translation("Foremax fibrillation/flutter right electrical axis low qrs amplitudes in limb leads moderate amplitude crit. For left ventricular hypertrophy aberrant qrs(t) course. High lateral infarction SHOULD be considered t-CHANGE, as at. inferior myocardial affection")

    #
    # text = "Hello World!"
    # target_languages = ["en", "tr", "de", "es", "it", "el", "zh", "ja", "ko"]
    #
    # print(f" {text} ".center(50, "-"))
    # for target_language in target_languages:
    #     translation = translate_text([text], target_language)
    #     source_language = translation.detected_language_code
    #     translated_text = translation.translated_text
    #     print(f"{source_language} → {target_language} : {translated_text}")
