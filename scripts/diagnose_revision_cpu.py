"""Report strict restriction gates without changing their return values.

Diagnostic only: original gates and exceptions are preserved. This probe also
compares source/target lifting on contiguous and strided copies of the same
input, to investigate CPU-backend differences. It is not an acceptance test.
"""
from __future__ import annotations
import importlib.util
import json
from pathlib import Path
import platform
import torch
from cdcureno.models import checkpoint_inflation as noncausal
from cdcureno.models import causal_checkpoint_inflation as causal


def load_fixture(name):
    path = Path(__file__).resolve().parents[1] / 'tests' / 'unit' / name
    spec = importlib.util.spec_from_file_location(path.stem, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def capture(module, name):
    original = getattr(module, name)
    def report(*args, **kwargs):
        result = original(*args, **kwargs)
        print(json.dumps({'gate': name, 'result': result}), flush=True)
        return result
    setattr(module, name, report)


print(json.dumps({'python': platform.python_version(), 'torch': torch.__version__,
                  'threads': torch.get_num_threads(), 'config': torch.__config__.show()}), flush=True)
capture(noncausal, 'verify_restriction_preservation')
capture(causal, 'verify_causal_models_on_input')
capture(causal, 'verify_target_future_invariance')
for name, filename in [('noncausal', 'test_checkpoint_inflation.py'),
                       ('causal', 'test_causal_checkpoint_inflation.py')]:
    fixture = load_fixture(filename)
    try:
        if name == 'noncausal':
            noncausal.inflate_checkpoint_payload(fixture._source_checkpoint(), fixture._target_config(), seed=44)
        else:
            fixture._inflate()
        print(json.dumps({'family': name, 'inflation': 'passed'}), flush=True)
    except Exception as error:
        print(json.dumps({'family': name, 'inflation': 'failed', 'error': repr(error)}), flush=True)

# A single linear operation with exactly equal values but different strides.
torch.manual_seed(9021)
linear = torch.nn.Linear(14, 6)
x = torch.randn(2, 7, 4, 14)
joined = torch.cat((x[..., None, :], torch.zeros(2, 7, 4, 1, 6)), dim=-1)
view = joined[..., :14]
with torch.no_grad():
    reference = linear(x)
    strided = linear(view).squeeze(3)
    contiguous = linear(view.contiguous()).squeeze(3)
print(json.dumps({'linear_stride_probe': {
    'strided_bitwise': torch.equal(reference, strided),
    'contiguous_bitwise': torch.equal(reference, contiguous),
    'strided_maximum_difference': float((reference-strided).abs().max()),
    'contiguous_maximum_difference': float((reference-contiguous).abs().max())}}), flush=True)
