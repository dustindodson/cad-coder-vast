#!/usr/bin/env python3
"""Image -> CadQuery Python backend for Qwen3-VL. Does not execute generated code.

Usage:
  python qwen_cad_backend.py --image input.png --output generated.py --max-new-tokens 24576

Run CadQuery execution separately in the isolated 'cad' environment.
"""
import argparse
import json
import os
import re
from pathlib import Path

PROMPT = '''You are an expert mechanical CAD programmer. Analyze the attached reference image and write a complete executable CadQuery Python script that models the visible object as a mechanically plausible 3D solid.
Rules:
- Output only Python code; do not use Markdown fences or explanatory prose.
- Use import cadquery as cq.
- Construct the object through CadQuery Workplane and standard CadQuery operations.
- Finish with a top-level variable named result holding the final solid or assembly.
- Do not import unavailable packages; do not require external assets or interactive UI.
- Favor clean, watertight solids and robust operations over speculative intricate details.
- Assume dimensions in millimeters. State assumptions as Python comments.
- Do not call exporters in the model script; the separate exporter handles STEP.
- Write complete code; do not truncate in the middle of a statement.
'''

def extract_python(output: str) -> str:
    match = re.search(r'```(?:python)?\s*\n(.*?)```', output, re.DOTALL | re.IGNORECASE)
    if match:
        return match.group(1).strip() + '\n'
    return output.strip() + '\n'

def generate(image: Path, tokens: int, model_id: str, prompt: str) -> tuple[str, dict]:
    import torch
    from transformers import AutoProcessor, Qwen3VLForConditionalGeneration
    if not torch.cuda.is_available():
        raise RuntimeError('CUDA GPU is required for this configuration')
    processor = AutoProcessor.from_pretrained(model_id)
    model = Qwen3VLForConditionalGeneration.from_pretrained(
        model_id, torch_dtype=torch.bfloat16, device_map='auto', attn_implementation='sdpa'
    )
    model.eval()
    messages = [{'role': 'user', 'content': [
        {'type': 'image', 'image': str(image.resolve())},
        {'type': 'text', 'text': prompt},
    ]}]
    inputs = processor.apply_chat_template(
        messages, tokenize=True, add_generation_prompt=True,
        return_dict=True, return_tensors='pt'
    ).to(model.device)
    n_input = inputs['input_ids'].shape[-1]
    max_context = getattr(model.config.text_config, 'max_position_embeddings', 262144)
    if n_input + tokens > max_context:
        raise ValueError(f'Input ({n_input}) + requested output ({tokens}) exceed context ({max_context})')
    with torch.inference_mode():
        generated = model.generate(
            **inputs, max_new_tokens=tokens, do_sample=False,
            use_cache=True, pad_token_id=processor.tokenizer.eos_token_id,
        )
    completion = generated[0][n_input:]
    raw = processor.decode(completion, skip_special_tokens=True, clean_up_tokenization_spaces=False)
    output_count = int(completion.numel())
    info = {
        'model': model_id,
        'requested_max_new_tokens': tokens,
        'input_tokens': int(n_input),
        'output_tokens': output_count,
        'possible_truncation': output_count >= tokens,
        'gpu': torch.cuda.get_device_name(0),
    }
    return extract_python(raw), info

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--image', type=Path, required=True)
    parser.add_argument('--output', type=Path, default=Path('generated_cad.py'))
    parser.add_argument('--max-new-tokens', type=int, default=24576)
    parser.add_argument('--model', default=os.environ.get('CAD_VL_MODEL', 'Qwen/Qwen3-VL-8B-Instruct'))
    parser.add_argument('--prompt-file', type=Path)
    args = parser.parse_args()
    if not args.image.is_file():
        parser.error(f'Image does not exist: {args.image}')
    if not 1 <= args.max_new_tokens <= 65536:
        parser.error('--max-new-tokens must be between 1 and 65536')
    prompt = args.prompt_file.read_text() if args.prompt_file else PROMPT
    code, info = generate(args.image, args.max_new_tokens, args.model, prompt)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(code, encoding='utf-8')
    try:
        compile(code, str(args.output), 'exec')
        info['python_syntax_valid'] = True
    except SyntaxError as exc:
        info['python_syntax_valid'] = False
        info['syntax_error'] = str(exc)
    report = args.output.with_suffix('.json')
    report.write_text(json.dumps(info, indent=2), encoding='utf-8')
    print(json.dumps({'python_file': str(args.output), 'report_file': str(report), **info}, indent=2))

if __name__ == '__main__':
    main()
