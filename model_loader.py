import torch
import timm

def load_model(model_path, num_classes):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = timm.create_model(
        "convnextv2_tiny",
        pretrained=False,
        num_classes=num_classes
    )

    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()

    return model, device
