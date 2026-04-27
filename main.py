import runpy
import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

print("STEP 1: Preprocessing")
runpy.run_path(os.path.join(BASE_DIR, "preprocessing.py"))

print("STEP 2: Analytics")
runpy.run_path(os.path.join(BASE_DIR, "analytic.py"))

print("STEP 3: Architecture")
runpy.run_path(os.path.join(BASE_DIR, "architecture.py"))

print("ALL DONE - check the output/ folder")
