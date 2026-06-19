# NanoDet-M 900k ZCU102/DPU Notes

This repo includes a ZCU102-oriented NanoDet-M 416 path for training a
lightweight 900k-class baseline and exporting raw DPU tensors.

## What Must Be Changed

1. Use NanoDet-M, not NanoDet-Plus-M.
   - NanoDet-M 416 is about 0.95M params.
   - NanoDet-Plus-M checkpoint previously tested is about 4.16M params and is
     not a valid 900k baseline.

2. Train/export with ReLU.
   - The stock legacy config uses `LeakyReLU`.
   - For this project, ReLU is safer for Vitis AI/DPU because earlier ZCU102
     compiler runs moved unsupported activations to CPU subgraphs.
   - Correct approach: train NanoDet-M again with `activation: ReLU`.
   - Quick compile-only test can load old weights with ReLU config, but accuracy
     is not report-valid.

3. Export raw DPU outputs.
   - Stock `tools/export_onnx.py` adds sigmoid, flatten, permute, and concat.
   - For ZCU102, keep DPU graph as raw per-scale tensors and do decode/NMS on
     ARM or HLS.
   - `export_nanodet_raw_onnx.py` exports:
     - `p3_raw`: `[1, 33, 52, 52]`
     - `p4_raw`: `[1, 33, 26, 26]`
     - `p5_raw`: `[1, 33, 13, 13]`
   - `33 = nc + 4 * (reg_max + 1) = 1 + 4 * 8`.

4. Expect possible CPU subgraphs from ShuffleNetV2 channel operations.
   - ShuffleNetV2 uses `chunk`, `cat`, `view`, `transpose`, and reshape for
     channel shuffle.
   - If Vitis AI compiler splits those to CPU, the stronger fix is replacing the
     backbone with a DPU-friendlier MobileNetV2/YOLO-style backbone or adding a
     custom channel-shuffle-free variant.

## Files

```text
config/nanodet_m_416_relu_raw_zcu.yml
tools/export_nanodet_raw_onnx.py
```

## Train Command

On Kaggle or another training machine, after preparing the YOLO-format dataset
at `/kaggle/working/pdt_nanodet_merge/{train,val,test}`:

```bash
python tools/train.py config/nanodet_m_416_relu_raw_zcu.yml
```

The expected best checkpoint is under:

```text
workspace/nanodet_m_416_relu_zcu/model_best/
```

## Export Command

Run after training a ReLU NanoDet-M checkpoint:

```bash
python tools/export_nanodet_raw_onnx.py \
  --cfg config/nanodet_m_416_relu_raw_zcu.yml \
  --weights workspace/nanodet_m_416_relu_zcu/model_best/model_best.ckpt \
  --out workspace/nanodet_m_416_relu_zcu/nanodet_m_416_relu_raw.onnx \
  --input-shape 416,416
```

Then quantize/compile this raw ONNX with Vitis AI. Postprocess must decode
NanoDet distribution boxes and NMS outside DPU.
