import importlib

def get_dataset(dataset_name):
    try:
        module = importlib.import_module(f"src.data.{dataset_name.lower()}_data")
        load_data = getattr(module, f"load_{dataset_name.lower()}_data")
        num_classes = getattr(module, "NUM_CLASSES")
        input_size = getattr(module, "INPUT_SIZE")
        return load_data, num_classes, input_size
    except ImportError:
        raise ValueError(f"Invalid dataset name: {dataset_name}")