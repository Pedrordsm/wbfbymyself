import pandas as pd
import csv

df = pd.read_csv('csvs/annotations_train.csv')

for rad_id, grupo in df.groupby('rad_id'):
    nome_arquivo = f"{rad_id}.txt"
    grupo[['image_id', 'class_name', 'x_min', 'y_min', 'x_max', 'y_max']].to_csv(
        nome_arquivo, 
        header=False, 
        index=False, 
        sep=',', 
        quoting=csv.QUOTE_NONE, 

    )