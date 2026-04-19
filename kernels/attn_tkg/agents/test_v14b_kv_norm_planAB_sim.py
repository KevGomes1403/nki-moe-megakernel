"""
Wrapper test for v14b_kv_norm_planAB — identical to test_v14b_kv_norm_sim.py
except the import points to v14b_kv_norm_planAB instead of v14b_kv_norm.
Do NOT modify test_v14b_kv_norm_sim.py.
"""

# Re-use all logic from the original test, but override the import
import importlib, sys, types

# Patch the import before the test module loads it
import v14b_kv_norm_planAB as _planAB_mod
# Register it under the name the test expects
sys.modules['v14b_kv_norm'] = _planAB_mod

# Now load and run the original test module
import test_v14b_kv_norm_sim as _test

if __name__ == "__main__":
    _test.main()
