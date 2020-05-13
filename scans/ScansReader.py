import SimpleITK as sitk
import numpy as np
import os
from .CoordsConverter import CoordsConverter
from .Scan import Scan
from .PatientInfoProvider import PatientInfoProvider


class ScansReader(object):
    def __init__(self, dir: str, patient_info_provider: PatientInfoProvider):
        self.dir = dir
        self.patient_info_provider = patient_info_provider

    def load_scan(self, path: str, file_name: str) -> Scan:
        scan_data = sitk.ReadImage(os.path.join(path, file_name))
        origin = np.array(scan_data.GetOrigin())
        spacing = np.array(scan_data.GetSpacing())
        name = file_name[:-4]

        return Scan(
            name,
            sitk.GetArrayFromImage(scan_data),
            CoordsConverter(origin, spacing),
            self.patient_info_provider.get_info(name),
        )

    def load(self):
        file_names = list(filter(lambda x: x.endswith(".mhd"), os.listdir(self.dir)))
        for file_name in file_names:
            yield self.load_scan(self.dir, file_name)
