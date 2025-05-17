from torchvision import transforms

# Define available transformations
available_transformations = {
    "ToTensor": transforms.ToTensor,
    "Normalize": lambda mean, std: transforms.Normalize(mean, std),
    "Resize": transforms.Resize
}

def get_transformation(transform_config):
    transform_name = transform_config["name"]
    if transform_name in available_transformations:
        if transform_name == "Normalize":
            mean = transform_config.get("mean", (0.1307,))
            std = transform_config.get("std", (0.3081,))
            return available_transformations[transform_name](mean, std)
        elif transform_name == "Resize":
            size = transform_config.get("size", (28, 28))
            return available_transformations[transform_name](size)
        else:
            return available_transformations[transform_name]()
    else:
        raise ValueError(f"Invalid transformation: {transform_name}")