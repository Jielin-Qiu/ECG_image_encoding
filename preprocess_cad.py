#!/usr/bin/env python
# coding: utf-8

import pandas as pd
import os
import subprocess
import shutil
import hashlib
from tqdm import tqdm


def decode_cad(src, des, use_hash_file_name=False):
    labels = os.listdir(src)
    labels = list(filter(lambda x: not x.startswith('.'), labels))
    labels.sort()

    os.makedirs(os.path.join(des, 'csv'), exist_ok=True)
    os.makedirs(os.path.join(des, 'xml'), exist_ok=True)

    df = pd.DataFrame(columns=['file_name', 'label', 'src'])

    for label in labels:
        for file in tqdm(os.listdir(os.path.join(src, label))):
            if not file.lower().endswith('.xml'):
                continue

            # decode xml file to csv file
            src_xml_file = os.path.join(src, label, file)
            subprocess.run(f"python musexmlex.py {src_xml_file} -e ISO-8859-1".split(" "), stdout=subprocess.DEVNULL)

            # compute hash value for decoded files
            tmp_csv_file = file[:-4] + '.csv'
            if not os.path.isfile(tmp_csv_file):
                print(f"can not find output file {tmp_csv_file}")
                continue

            with open(tmp_csv_file) as f:
                text = f.read()

            if use_hash_file_name:
                hash_file_name = hashlib.md5(text.encode()).hexdigest().upper()
            else:
                hash_file_name = file[:file.index('.')]

            # move decoded csv file to destination
            des_csv_file = os.path.join(des, 'csv', hash_file_name + '.csv')
            os.replace(tmp_csv_file, des_csv_file)

            # copy original xml file to destination
            des_xml_file = os.path.join(des, 'xml', hash_file_name + '.xml')
            if os.path.isfile(des_xml_file):
                print(des_xml_file, "file already exist.")
            shutil.copyfile(src_xml_file, des_xml_file)

            # add file name and labels to the dataframe
            df_tmp = pd.DataFrame({'file_name': [hash_file_name], 'label': [label], 'src': [src_xml_file]})
            df = pd.concat([df, df_tmp])

    # update label file
    label_path = os.path.join(des, 'label.csv')
    if os.path.exists(label_path):
        df_old = pd.read_csv(label_path)
        df = pd.concat([df_old, df])

    # drop duplicates
    df = df.drop_duplicates(subset=['file_name'])
    df = df.sort_values(by=['file_name'])
    df.to_csv(label_path, index=False)


def main():
    # decode_cad('data/original/CAD_20221103', 'data/decoded/CAD')
    # decode_cad('data/original/CAD_20221210', 'data/decoded/CAD')
    decode_cad('data/original/CAD_20230315', 'data/decoded/CAD')


if __name__ == '__main__':
    main()
