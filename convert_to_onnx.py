import torch.onnx
import onnx

import Models

def get_cfg():
    cfg={
            "input_type":"rgb",
            "mod":"mod",
            "thres":0 # No need?
        }
    cfg["channels_in"] = 1 if cfg["input_type"] == 'depth' else 4
    return cfg

def get_model(cfg):
    # Init model
    model = Models.define_model(mod=cfg["mod"], in_channels=cfg["channels_in"], thres=cfg["thres"])
    return model

def load_checkpoints(model,ckpt_path='Checkpoints/FusionNet_Depth_Completion/model_best_epoch.pth.tar'):
    checkpoint = torch.load(ckpt_path)
    model.load_state_dict(checkpoint['state_dict'])

    # set the model to inference mode
    model.eval()

    return model

def export_onnx(model,cfg, onnx_path="FusionNet.onnx"):
    
    # Input to the model
    batch_size = 1
    H, W = 256, 1216
    x = torch.randn(batch_size, cfg["channels_in"], H, W, requires_grad=True)

    torch_out = model(x)

    # Export the model
    torch.onnx.export(model,               # model being run
                    x,                         # model input (or a tuple for multiple inputs)
                    onnx_path,   # where to save the model (can be a file or file-like object)
                    export_params=True,        # store the trained parameter weights inside the model file
                    opset_version=10,          # the ONNX version to export the model to
                    do_constant_folding=True,  # whether to execute constant folding for optimization
                    input_names = ['input'],   # the model's input names
                    output_names = ['output'], # the model's output names
                    dynamic_axes={'input' : {0 : 'batch_size'},    # variable length axes
                                    'output' : {0 : 'batch_size'}})

def verify_onnx(onnx_path):
    onnx_model = onnx.load(onnx_path)
    onnx.checker.check_model(onnx_model)

if __name__== '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='Convert Pytorch Model into ONNX Format')
    parser.add_argument('--onnx_path', default='Out/FusionNet.onnx', help='ONNX file save path')
    parser.add_argument('--ckpt_path', default='Checkpoints/model_best_epoch.pth.tar', help='Path to checkpoint file')
    args = parser.parse_args()


    ckpt_path = args.ckpt_path
    onnx_path = args.onnx_path
    
    # Get model config
    cfg = get_cfg()
    # Get model definition
    model = get_model(cfg)

    # Initialize model with the pretrained weights
    model = load_checkpoints(model,ckpt_path)
    
    # Export to ONNX
    export_onnx(model,cfg,onnx_path)

    # Check model
    verify_onnx(onnx_path)