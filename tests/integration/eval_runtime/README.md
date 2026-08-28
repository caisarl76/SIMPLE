# Dedicated PSI0 runtime integration gates

These tests are never collected by `pytest tests/eval_runtime`.
Run only the exact gate command named in the implementation plan. Gates 0--6
must use fakes or model-free fixtures. Gates 7--9 require explicit operator
approval, VPN/SSH availability, an approved profile commit, PC2 GPU 1, and
H100 GPU 7. No gate authorizes real robot control.
