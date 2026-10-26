# CAD-Coder replacement: Qwen3-VL

This is a testable **alternative vision-model backend**, not a proven image-to-CAD specialist. It creates a CadQuery script from an image, with a configurable generation budget, and keeps STEP conversion separate. Existing Gradio integration is not replaced here, because the project's current `web_gui.py` isn't available for inspection.

## On Vast.ai

Use a CUDA/PyTorch environment compatible with the RTX PRO 5000 Blackwell (`sm_120`). Do not reuse the original PyTorch 2.1.2+cu121 environment. Confirm support with:

```bash
python -c 'import torch; print(torch.__version__,torch.version.cuda,torch.cuda.get_device_name(0)); print(torch.cuda.get_arch_list())'
```

Install packages inside that working model environment:

```bash
python -m pip install -U 'transformers>=4.57,<6' accelerate pillow safetensors qwen-vl-utils
```

Start with a **4,096-token smoke test** before requesting 24,576:

```bash
python qwen_cad_backend.py --image /opt/app/test.png --output /opt/app/generated_cad.py --max-new-tokens 4096
python qwen_cad_backend.py --image /opt/app/test.png --output /opt/app/generated_cad.py --max-new-tokens 24576
```

The `.json` report records actual generated tokens and whether the output reached the requested cap. Outputs can end earlier if the model emits EOS. 24,576 is an available **requested** budget, not a guarantee of that output length or quality.

## Export STEP

Review the generated Python before execution. It is untrusted, runnable code. Execute in a **sandboxed, restricted** CAD container with no secrets, network access, or sensitive volumes, using the CadQuery environment:

```bash
/opt/conda/envs/cad/bin/python export_step.py generated_cad.py generated.step
```

The exporter assumes `/opt/conda/envs/cad` already has a working `cadquery`/OCC installation including needed OSMesa libraries.

## Integration into existing Gradio app

Replace the old LLaVA inference call with `generate(image_path, tokens, model_id, PROMPT)` from `qwen_cad_backend.py`. Keep the existing image uploader and STEP-download UI. Model loading is expensive; **refactor generate() to cache processor/model at process scope before wiring requests**. This standalone script loads them per invocation deliberately to keep the first test simple.

## Follow-up validation

1. Verify PyTorch `sm_120` support.
2. Check 4K and 24K generation, actual counts and truncation.
3. Run syntax check then STEP export on a known simple reference.
4. Compare resulting CAD geometry to image; improve prompt and add a repair pass.
5. Only then swap into Gradio and evaluate a quantized 32B variant.
