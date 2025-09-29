import pandas as pd
import os
import sys
import yaml
from sklearn.preprocessing import StandardScaler

params = yaml.safe_load(open("params.yaml"))['preprocess']

def preprocess(input_path, output_path):
    df = pd.read_csv(input_path)
    std = StandardScaler()
    df[df.columns[:8]] = df[df.columns[:8]].astype(float)
    df.iloc[:,:8] = std.fit_transform(df.iloc[:,:8])
    df = pd.DataFrame(df, columns=df.columns)
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    df.to_csv(output_path, header=None, index=None)
    print(f"Preprocees data saved to {output_path}")

if __name__=="__main__":
    preprocess(params['input'], params['output'])



