from .CoordsConverter import CoordsConverter
from preprocess import normalize_scan_image


def get_coords_from_node_row(row):
    return row[["coordX", "coordY", "coordZ"]]


class Scan(object):
    def __init__(
        self, name: str, images, converter: CoordsConverter, suspicious_places
    ):
        self.name = name
        self.images = images
        self.converter = converter
        self.suspicious_places = suspicious_places

    def get_name(self):
        return self.name

    def get_converter(self):
        return self.converter

    def get_images(self):
        return self.images

    def crop_subimage(self, coords, width=50):
        x, y, z = self.converter.convert_to_voxel_coords(coords)
        return self.images[
            int(z),
            int(y - width / 2) : int(y + width / 2),
            int(x - width / 2) : int(x + width / 2),
        ]

    def get_suspicious_places(self):
        return self.suspicious_places

    def is_empty(self) -> bool:
        return len(self.suspicious_places) == 0

    def size(self) -> int:
        return len(self.suspicious_places)

    def extract_images(self, preprocess_function=normalize_scan_image):
        for _, row in self.suspicious_places.iterrows():
            yield normalize_scan_image(
                self.crop_subimage(get_coords_from_node_row(row))
            ), row["class"]
