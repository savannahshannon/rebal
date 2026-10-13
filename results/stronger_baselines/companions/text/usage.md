# Text execution and RNG compatibility companion

These files accompany the immutable Text numeric evidence archive. They are kept under `companions/text/`, outside the raw `text/` manifest and its exact directory. The numeric evidence archive remains usable for the weight-free independent verification described in its own documentation.

The compatibility adapter handles one checkpoint-loading issue in the unchanged scientific source: mapping a checkpoint to MPS also maps its saved MPS RNG byte tensor to MPS, while the restore function requires CPU byte storage. The adapter makes a shallow copy of the RNG record and moves only that byte tensor to CPU before calling the frozen restore function. It does not replace model loading, model forward functions, optimization, calibration, metrics, selections, or checkpoint bytes. It accepts only the exact prescribed MPS evaluation arguments and rejects training or changed settings before importing the frozen runner.

The adapter is tied to the original project layout and fixed arguments. To inspect or rerun its eight isolated fixtures, restore `checkpoint_rng_cpu_decode_compatibility.py` and `test_checkpoint_rng_cpu_decode_compatibility.py` to `work/next_round/text/` in that layout. Run the tests with the original environment from the project root:

```sh
work/venv/bin/python -m unittest discover -s work/next_round/text -p test_checkpoint_rng_cpu_decode_compatibility.py -v
```

Running the full evaluation additionally requires the original model assets, prepared text inputs, token/cache arrays and trained checkpoints. Those assets are deliberately omitted from the numeric archive. An invocation from the archive's `companions/text/` location is not a portable full-model evaluator.

With all original assets restored and the completed training identities independently validated, the exact original evaluation invocation can be obtained from the adapter's immutable argument tuple. It must be launched from the original project root:

```python
import importlib.util
import subprocess
from pathlib import Path
path = Path("work/next_round/text/checkpoint_rng_cpu_decode_compatibility.py")
spec = importlib.util.spec_from_file_location("text_rng_compatibility", path)
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)
subprocess.run(["work/venv/bin/python", "-u", str(path), *module.FROZEN_MPS_EVALUATION_ARGV], check=True)
```

The saved execution receipt records the successful CPU, recovered MPS and aggregate-verification phases separately from the original failed MPS attempt. That attempt produced zero MPS confirmation cells. The isolated generator receipt concerns random-generator byte storage only; it loaded no production model, read no source rows and produced no confirmation predictions. Independent numeric verification and original actual-state verification have separate scopes and receipts.
