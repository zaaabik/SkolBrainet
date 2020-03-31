import argparse
import os
import re

import pandas as pd

DOMAIN_REG_EP = r'_(.*?_\d+)_'


def create_labels(img_path, out):
    files = os.listdir(img_path)

    domains = []
    for file in files:
        domains.append(
            re.findall(DOMAIN_REG_EP, file)[0]
        )

    labels_df = pd.DataFrame(
        {
            'Filename': files,
            'Domain': domains
        }
    )
    out_path = os.path.join(out, 'labels.csv')
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
