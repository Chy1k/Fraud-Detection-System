import pandas as pd

def load_data():
    data = pd.read_csv('creditcard.csv')
    return data