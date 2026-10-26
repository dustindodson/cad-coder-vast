#!/usr/bin/env python3
"""Run only trusted/reviewed generated Python! CAD scripts execute arbitrary code.
Run with the dedicated CadQuery interpreter/environment, not the model environment.
Usage: /opt/conda/envs/cad/bin/python export_step.py generated_cad.py output.step
"""
import runpy
import sys
from pathlib import Path
import cadquery as cq

if len(sys.argv) != 3:
    raise SystemExit('usage: export_step.py <generated.py> <output.step>')
source = Path(sys.argv[1]).resolve()
target = Path(sys.argv[2]).resolve()
ns = runpy.run_path(str(source), run_name='__cad_model__')
result = ns.get('result')
if result is None:
    raise ValueError("Generated Python did not define variable 'result'")
cq.exporters.export(result, str(target))
print(f'STEP exported: {target}')
