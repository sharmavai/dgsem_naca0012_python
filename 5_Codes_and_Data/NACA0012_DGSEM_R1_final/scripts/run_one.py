"""Run a single case: resumes its own checkpoint if present, otherwise
starts from the potential-flow (panel) initial condition."""
import sys, json, time
import numpy as np
sys.path.insert(0, 'src')
sys.path.insert(0, 'scripts')
from run_production import run_case, MESH_MEDIUM, case_label, RESULTS

Mach = float(sys.argv[1]); alpha = float(sys.argv[2])
maxwall = float(sys.argv[3]) if len(sys.argv) > 3 else 520.0
mesh_tag = sys.argv[4] if len(sys.argv) > 4 else ""
shock = (len(sys.argv) > 5 and sys.argv[5] == 'shock')
tag = case_label(Mach, alpha, mesh_tag=mesh_tag)
ok, U, info = run_case(Mach, alpha, MESH_MEDIUM, tag, U_init=None,
                       max_wall_s=maxwall, shock=shock)
print(json.dumps({k: v for k, v in info.items() if k != 'U'}))
