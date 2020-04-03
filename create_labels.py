import argparse
import os
import re
import numpy as np
import pandas as pd

DOMAIN_REG_EP = r'_(.*?_\d+)_'

max_idx = 169
min_idx = 120
index_reg_exp = 'CC(.*?)_'


def is_train(file_name):
    index = re.findall(index_reg_exp, file_name)[0]
    index = int(index)
    if min_idx <= index <= max_idx:
        return True
    return False


def inTestSet (name):
    # Here I specify the test domain
    N = int(name[2:6])
    Ph15_t = np.arange(50,60)
    Ph3_t = np.arange(110,120)
    S15_t = np.arange(170,180)
    S3_t = np.arange(230,240)
    GE15_t = np.arange(290,300)
    GE3_t = np.arange(350, 360)
    test_scans = np.concatenate((Ph15_t, Ph3_t, S15_t, S3_t, GE15_t, GE3_t))
    if N in test_scans:
        return True
    else:
        return False

def trueStep (name):



def create_labels(img_path, out):
    files_all = os.listdir(img_path)
    files_train = []
    domains = []
    for file in files_all:
        if not inTestSet(file):
            files_train.append(file)
            domains.append(
                re.findall(DOMAIN_REG_EP, file)[0]
            )

    labels_df = pd.DataFrame(
        {
            'Filename': files_train,
            'Domain': domains
        }
    )
    labels_df = pd.DataFrame.sort_values(labels_df, by='Filename')
    labels_df['Labeled'] = labels_df['Filename'].apply(is_train)

    out_path = os.path.join(out, 'labels_old.csv')
    labels_df.to_csv(out_path, index=None)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Script creates domain labels csv file for MRI dataset')
    parser.add_argument('--path', type=str, help='Path to folder with images')
    parser.add_argument('--out', type=str, help='Path for result csv file')

    args = parser.parse_args()
    path = args.path
    out = args.out

    create_labels(path, out)
