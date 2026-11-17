"""Run without GPU fitting: geometry invariance and invalid-data rejection."""
import importlib.util
from pathlib import Path
import numpy as np

path=Path(__file__).with_name('combined_training.py')
spec=importlib.util.spec_from_file_location('combined',path)
module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
rng=np.random.default_rng(42)
x=rng.normal(size=(32,6,3))
a=np.array(list(module.common_movement(x).values()))
rotation=np.linalg.qr(rng.normal(size=(3,3)))[0]
b=np.array(list(module.common_movement((x@rotation)*2.7+np.array([4,5,-2])).values()))
assert a.shape==(45,) and np.allclose(a,b,rtol=1e-9,atol=1e-9)
for bad in [np.zeros((5,6,3)),np.zeros((2,6,3)),np.full((5,6,3),np.nan)]:
    try: module.common_movement(bad)
    except ValueError: pass
    else: raise AssertionError('Invalid geometry accepted')
print('PASS: 45 shared features invariant to scale/rotation/translation; invalid skeletons rejected')
