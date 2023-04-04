import torch
import intel_pytorch_extension as ipex

from unet3d_kits_model import Unet3D

def trace_and_save(model, save_path="unet3d_jit_model.pt"):
    print("Start trace model")
        
    with torch.no_grad():
        image = torch.rand([1, 1, 128, 128, 128]).to(ipex.DEVICE)
        model = torch.jit.trace(model, (image), check_trace=False)
        model = torch.jit.freeze(model)
        torch.jit.save(model.to("cpu"), save_path)

    print("Finish trace")

def load_model(model_path):  
    model = Unet3D()
    checkpoint = torch.load(model_path, map_location="cpu")
    model.load_state_dict(checkpoint['best_model_state_dict'])
    model = model.to(ipex.DEVICE)
    model.eval()
    trace_and_save(model)

if __name__ == "__main__":
    load_model(model_path="build/model/3dunet_kits19_pytorch_checkpoint.pth")

