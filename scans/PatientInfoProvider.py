import pandas as pd


class PatientInfoProvider(object):
    def __init__(self, path: str):
        self.df = pd.read_csv(path)

    def get_info(self, name: str):
        return self.df[self.df["seriesuid"] == name]
