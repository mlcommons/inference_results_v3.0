import torch
import torch.fx.experimental.optimization as optimization
import intel_extension_for_pytorch as ipex
import torch.nn as nn

class GC_RN50_Start(torch.nn.Module):

    def __init__(self, rn50, scale_out: float=1.0, zero_out: int=0, skip_quant: bool=True, scale_in: float=1.0, zero_in: int=0, skip_quant_in: bool=True):
        super(GC_RN50_Start, self).__init__()
        
        rn50_start = torch.nn.Sequential(rn50.conv1, rn50.bn1, rn50.relu, rn50.maxpool)
        rn50_start.eval()

        with torch.no_grad():
            rn50_start = optimization.fuse(rn50_start, inplace=True)

        self.start = rn50_start

        self.scale_out = scale_out
        self.zero_out = zero_out
        self.scale_in = scale_in
        self.zero_in = zero_in
        self.skip_quant=skip_quant
        self.skip_quant_in=skip_quant_in

    def forward(self, x):
        x = self.start(x)
        if not self.skip_quant_in:
            x = x.to(torch.int8)
            x = torch._make_per_tensor_quantized_tensor(x, self.scale_in, self.zero_in)
        if not self.skip_quant:
            # x = self.quant(x)
            x = torch.quantize_per_tensor(x, self.scale_out, self.zero_out, torch.qint8).int_repr()
        return x


class GC_RN50_Middle(torch.nn.Module):

    def __init__(self, rn50, scale_out: float=1.0, zero_out: int=0, skip_quant_in: bool=True, skip_quant_out: bool=True):
        super(GC_RN50_Middle, self).__init__()

        rn50.eval()
        
        self.layer1 = rn50.layer1
        self.layer2 = rn50.layer2
        self.layer3 = rn50.layer3
        self.layer4 = rn50.layer4

        self.scale_out = scale_out
        self.zero_out = zero_out
        self.skip_quant_in = skip_quant_in
        self.skip_quant_out = skip_quant_out
        self.dequant = torch.quantization.DeQuantStub()
        self.quant = torch.quantization.QuantStub()

    def forward(self, x):

        if not self.skip_quant_in:
            x = torch.dequantize(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        if not self.skip_quant_out:
            x = torch.quantize_per_tensor(x, self.scale_out, self.zero_out, torch.qint8).int_repr()

        return x


class GC_RN50_End(torch.nn.Module):

    def __init__(self, rn50, scale_in: float=1.0, zero_in: int=0, skip_quant: bool=True):
        super(GC_RN50_End, self).__init__()

        self.avgpool = nn.AvgPool2d(7)
        self.fc = rn50.fc

        self.skip_quant = skip_quant
        self.scale_in = scale_in
        self.zero_in = zero_in
        self.dequant = torch.quantization.DeQuantStub()

    def forward(self, x):
        if not self.skip_quant:
            x = torch._make_per_tensor_quantized_tensor(x, self.scale_in, self.zero_in)
            x = torch.dequantize(x)
        x = self.avgpool(x)
        x = torch.flatten(x,1)
        x = self.fc(x)
        return x

