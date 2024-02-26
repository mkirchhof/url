from .reader_tfds import ReaderTfds, get_class_labels

class ReaderVTAB(ReaderTfds):
    def __init__(self, root, name, split, *args, **kwargs):
        # Match the name to the VTAB task
        if name == "caltech101":
            from .vtab.caltech import Caltech101
            tfds_obj = Caltech101(data_dir=root)
        elif name == "cars196":
            from .vtab.cars import CarsData
            tfds_obj = CarsData(data_dir=root)
        elif name == "cifar10":
            from .vtab.cifar import CifarData
            tfds_obj = CifarData(num_classes=10, data_dir=root)
        elif name == "cifar100":
            from .vtab.cifar import CifarData
            tfds_obj = CifarData(num_classes=100, data_dir=root)
        elif name == "clevr_count_all":
            from .vtab.clevr import CLEVRData
            tfds_obj = CLEVRData(task="count_all", data_dir=root)
            name = "clevr"
        elif name == "clevr_closest_object_distance":
            from .vtab.clevr import CLEVRData
            tfds_obj = CLEVRData(task="closest_object_distance", data_dir=root)
            name = "clevr"
        elif name == "cub200":
            from .vtab.cub import CUB2011Data
            tfds_obj = CUB2011Data(data_dir=root)
            name = "caltech_birds2011"
        elif name == "retinopathy":
            from .vtab.diabetic_retinopathy import RetinopathyData
            tfds_obj = RetinopathyData(data_dir=root)
            name = "diabetic_retinopathy_detection/btgraham-300"
        elif name == "dmlab":
            from .vtab.dmlab import DmlabData
            tfds_obj = DmlabData(data_dir=root)
        elif name == "dsprites_label_orientation":
            from .vtab.dsprites import DSpritesData
            tfds_obj = DSpritesData(predicted_attribute="label_orientation", num_classes=16, data_dir=root)
            name = "dsprites"
        elif name == "dsprites_label_x_position":
            from .vtab.dsprites import DSpritesData
            tfds_obj = DSpritesData(predicted_attribute="label_x_position", num_classes=16, data_dir=root)
            name = "dsprites"
        elif name == "dtd":
            from .vtab.dtd import DTDData
            tfds_obj = DTDData(data_dir=root)
        elif name == "eurosat":
            from .vtab.eurosat import EurosatData
            tfds_obj = EurosatData(data_dir=root)
            name = "eurosat/rgb"
        elif name == "food101":
            from .vtab.food101 import Food101Data
            tfds_obj = Food101Data(data_dir=root)
        elif name == "inaturalist":
            from .vtab.inaturalist import INaturalistData
            tfds_obj = INaturalistData(data_dir=root)
            name = "i_naturalist2017"
        elif name == "kitti":
            from .vtab.kitti import KittiData
            tfds_obj = KittiData(task="closest_vehicle_distance", data_dir=root)
        elif name == "oxford_flowers102":
            from .vtab.oxford_flowers102 import OxfordFlowers102Data
            tfds_obj = OxfordFlowers102Data(data_dir=root)
        elif name == "oxford_iiit_pet":
            from .vtab.oxford_iiit_pet import OxfordIIITPetData
            tfds_obj = OxfordIIITPetData(data_dir=root)
        elif name == "patch_camelyon":
            from .vtab.patch_camelyon import PatchCamelyonData
            tfds_obj = PatchCamelyonData(data_dir=root)
        elif name == "resisc45":
            from .vtab.resisc45 import Resisc45Data
            tfds_obj = Resisc45Data(data_dir=root)
        elif name == "smallnorb_label_azimuth":
            from .vtab.smallnorb import SmallNORBData
            tfds_obj = SmallNORBData(predicted_attribute="label_azimuth", data_dir=root)
            name = "smallnorb"
        elif name == "smallnorb_label_elevation":
            from .vtab.smallnorb import SmallNORBData
            tfds_obj = SmallNORBData(predicted_attribute="label_elevation", data_dir=root)
            name = "smallnorb"
        elif name == "sun397":
            from .vtab.sun397 import Sun397Data
            tfds_obj = Sun397Data(data_dir=root)
            name = "sun397/tfds"
        elif name == "svhn":
            from .vtab.svhn import SvhnData
            tfds_obj = SvhnData(data_dir=root)
            name = "svhn_cropped"
        else:
            raise NotImplementedError(f"VTAB task {name} is not implemented.")

        # The rest of this is just a default ReaderTFDS
        kwargs["download"] = False  # The readers above have already downloaded
        tfds_split = tfds_obj._tfds_splits[split]  # translate VTAB to TFDS split name
        super(ReaderVTAB, self).__init__(root=root,
                                         name=name,
                                         split=tfds_split,
                                         decoder=tfds_obj.composed_decoder,
                                         postprocess_fn=tfds_obj.postprocess_fn,
                                         target_name=tfds_obj.default_label_key,
                                         input_name=tfds_obj._image_key,
                                         *args,
                                         **kwargs)

        # The superclass ReaderTFDS loads the default dataset.
        # Replace it with the VTAB specific one we loaded above.
        self.builder = tfds_obj._dataset_builder
        self.class_to_idx = get_class_labels(self.builder.info) if self.target_name == "label" else {}
