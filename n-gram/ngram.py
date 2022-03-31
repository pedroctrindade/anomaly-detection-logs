import pandas as pd
from nltk.util import bigrams
from nltk import FreqDist
from sklearn.decomposition import PCA
import csv
import re
def n_gram_string(s, n=1):

    split_strings = [s[index: index + n] for index in range(0, len(s), n)]

    tokens = [token for token in split_strings if token != ""]
    bgs = bigrams(tokens)
    fdist = FreqDist(bgs)


    return fdist

'''
Create a list containing all features and 
create a new dict with string key instead of tuple key
'''
def parse_sequence(f_matrix_keys, sequence):
    new_sequence = {}
    for k, v in sequence.items():
        f_matrix_key = ''.join(k)
        new_sequence[f_matrix_key] = v
        if f_matrix_key not in f_matrix_keys:
            f_matrix_keys.append(f_matrix_key)

    return new_sequence

def write_dict_to_csv(keys, my_dictionary):

    with open('../../Dataset/trafficDataset.csv', 'w', newline='') as output_file:
        dict_writer = csv.DictWriter(output_file, keys, quoting=csv.QUOTE_ALL, restval=0)
        dict_writer.writeheader()
        dict_writer.writerows(my_dictionary)


    return
'''
Lowercase str and replace special char and numbers
'''
def format_str(str):
    str = re.sub(r'[^a-zA-Z0-9 \n\.]', '@', str)
    # Replace numbers with another custom char
    str = re.sub(r'[^a-zA-Z@ \n\.]', '#', str)
    return str.lower()


def main():

    df_normal = pd.read_json("../../Dataset/normalTrafficTraining.json")
    df_normal["REQUEST_TYPE"] = "Normal"
    df_anomalous = pd.read_json("../../Dataset/anomalousTrafficTest.json")
    df_anomalous["REQUEST_TYPE"] = "Anomalous"
    df = df_normal.append(df_anomalous)

    n_grams_seq = []

    f_matrix_keys = []

    for index,data in df.iterrows():

        feature_str = format_str(data.RESOURCE)

        if data.PAYLOAD != "":
            feature_str += format_str(data.PAYLOAD)

        sequence = n_gram_string(feature_str)
        parsed_sequence = parse_sequence(f_matrix_keys, sequence)
        parsed_sequence["REQUEST_TYPE"] = data.REQUEST_TYPE
        parsed_sequence["FULL_REQUEST"] = "{}?{}".format(data.RESOURCE, data.PAYLOAD)
        n_grams_seq.append(parsed_sequence)

    f_matrix_keys.append("REQUEST_TYPE")
    f_matrix_keys.append("FULL_REQUEST")
    write_dict_to_csv(f_matrix_keys, n_grams_seq)

    #no of features
    print(len(f_matrix_keys))

    return

if __name__ == "__main__":
    main()