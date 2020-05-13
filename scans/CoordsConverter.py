import numpy as np


class CoordsConverter(object):
    def __init__(self, origin, spacing):
        self.origin = origin
        self.spacing = spacing

    def convert_to_voxel_coords(self, coords):
        voxel_coords = [
            np.absolute(coords[i] - self.origin[i]) / self.spacing[i]
            for i in range(len(coords))
        ]
        return tuple(voxel_coords)
