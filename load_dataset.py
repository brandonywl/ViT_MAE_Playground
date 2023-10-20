from ZipDataset import create_dataloader


class DataloaderGenerator():
    base_data_path = "./data/"
    WBC_CLASSNAMES = ['Basophil', 'Monocyte', 'Eosinophil', 'Lymphocyte', 'Neutrophil']
    CAM16_CLASSNAMES = ['normal', 'tumor']
    PRCC_CLASSNAMES = None

    wbc100_datapath = base_data_path + "WBC_100.zip"
    wbc50_datapath = base_data_path + "WBC_50.zip"
    wbc10_datapath = base_data_path + "WBC_10.zip"
    wbc1_datapath = base_data_path + "WBC_1.zip"

    cam16_datapath = base_data_path + "CAM16_100cls_10mask.zip"
    prcc_datapath = base_data_path + "pRCC_nolabel.zip"

    dataset_to_path_map = {
        "wbc100": wbc100_datapath,
        "wbc50": wbc50_datapath,
        "wbc10": wbc10_datapath,
        "wbc1": wbc1_datapath,
        "cam16": cam16_datapath,
        "prcc": prcc_datapath
    }

    dataset_to_classname_map = {
        "wbc": WBC_CLASSNAMES,
        "cam16": CAM16_CLASSNAMES,
        "prcc": PRCC_CLASSNAMES
    }

    @staticmethod
    def get_dataset_types():
        return list(DataloaderGenerator.dataset_to_path_map.keys())
    
    @staticmethod
    def get_classnames(dataset):
        mapper = DataloaderGenerator.dataset_to_classname_map
        if dataset[:4].lower() == "prcc":
            return mapper["prcc"]
        elif dataset.lower() == "cam16":
            return mapper[dataset]
        else:
            return mapper["wbc"]

        
    @staticmethod
    def dataloader(dataset_type, batch_size, input_image_shape, subfolder="", classnames="", stain_normalize=False, get_mask=False):
        datapath = DataloaderGenerator.dataset_to_path_map[dataset_type.lower()]
        if classnames == "":
            classnames = DataloaderGenerator.get_classnames(dataset_type)

        return create_dataloader(datapath, subfolder, classnames=classnames, batch_size=batch_size, input_image_shape=input_image_shape, stain_normalize=stain_normalize, get_mask=get_mask)

