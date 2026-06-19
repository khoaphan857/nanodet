import argparse
import types
from pathlib import Path

import torch

from nanodet.model.arch import build_model
from nanodet.model.head.nanodet_head import NanoDetHead
from nanodet.util import Logger, cfg, load_config, load_model_weight


def raw_forward_onnx(self, feats):
    """Return raw per-scale NCHW tensors for DPU compilation.

    NanoDet's stock ONNX export applies sigmoid, flatten, permute, and concat.
    Those are better kept outside DPU for ZCU102, similar to DetectRaw.
    """
    outputs = []
    for x, cls_convs, reg_convs, gfl_cls, gfl_reg in zip(
        feats, self.cls_convs, self.reg_convs, self.gfl_cls, self.gfl_reg
    ):
        cls_feat = x
        reg_feat = x
        for cls_conv in cls_convs:
            cls_feat = cls_conv(cls_feat)
        for reg_conv in reg_convs:
            reg_feat = reg_conv(reg_feat)
        if self.share_cls_reg:
            out = gfl_cls(cls_feat)
        else:
            cls_pred = gfl_cls(cls_feat)
            reg_pred = gfl_reg(reg_feat)
            out = torch.cat([cls_pred, reg_pred], dim=1)
        outputs.append(out)
    return tuple(outputs)


def patch_raw_export(model):
    for module in model.modules():
        if isinstance(module, NanoDetHead):
            module._forward_onnx = types.MethodType(raw_forward_onnx, module)


def load_checkpoint(model, model_path, save_dir):
    logger = Logger(-1, save_dir, False)
    checkpoint = torch.load(model_path, map_location="cpu")
    load_model_weight(model, checkpoint, logger)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cfg", required=True)
    parser.add_argument("--weights", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--input-shape", default="416,416")
    args = parser.parse_args()

    h, w = [int(x) for x in args.input_shape.split(",")]
    load_config(cfg, args.cfg)
    model = build_model(cfg.model)
    load_checkpoint(model, args.weights, cfg.save_dir)
    patch_raw_export(model)
    model.eval()

    dummy = torch.randn(1, 3, h, w)
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    with torch.no_grad():
        ys = model(dummy)
    print("Raw output shapes:", [tuple(y.shape) for y in ys])

    torch.onnx.export(
        model,
        dummy,
        str(out_path),
        opset_version=11,
        input_names=["data"],
        output_names=["p3_raw", "p4_raw", "p5_raw"],
        do_constant_folding=True,
        keep_initializers_as_inputs=False,
    )
    print("Saved:", out_path)


if __name__ == "__main__":
    main()
