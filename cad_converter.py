"""Run by a low-privilege, resource-limited subprocess (not a full security sandbox).

Usage: cad_converter.py generated.py output.step preview.stl
"""
import runpy
import sys
from pathlib import Path

import cadquery as cq


def export_generated(source: Path, step: Path, stl: Path) -> None:
    # Run the generated script in the subprocess, never inside the Gradio server.
    values = runpy.run_path(str(source), run_name='__main__')
    candidates = []
    for key, value in values.items():
        if key.startswith('__'):
            continue
        if isinstance(value, (cq.Workplane, cq.Shape)):
            candidates.append((key, value))
    if not candidates:
        raise ValueError('No CadQuery Workplane or Shape found in generated code')
    # Prefer conventional names, else use the most recently assigned matching object.
    by_name = dict(candidates)
    model = next((by_name[key] for key in ('result', 'model', 'part', 'solid', 'shape', 'obj') if key in by_name), candidates[-1][1])
    solid = model.val() if isinstance(model, cq.Workplane) else model
    if solid.isNull():
        raise ValueError('Generated shape is null')
    cq.exporters.export(model, str(step), exportType='STEP')
    cq.exporters.export(model, str(stl), exportType='STL', tolerance=0.1, angularTolerance=0.2)
    if not step.is_file() or step.stat().st_size == 0:
        raise ValueError('STEP output was not created')


if __name__ == '__main__':
    if len(sys.argv) != 4:
        sys.exit('Usage: cad_converter.py input.py output.step output.stl')
    export_generated(*(Path(v) for v in sys.argv[1:]))
