import os
import re
import scipy.io
import matplotlib.pyplot as plt
import numpy as np

# Base directory structure
base_dir = 'output'
conditions = ['dox', 'no_dox']
time_step = 60  # in minutes (adjust based on your simulation)

for cond in conditions:
    condition_path = os.path.join(base_dir, cond)
    if not os.path.isdir(condition_path):
        continue

    print(f"\nProcessing {cond.upper()}...\n")

    for run_name in sorted(os.listdir(condition_path)):
        run_path = os.path.join(condition_path, run_name)
        if not os.path.isdir(run_path):
            continue

        print(f"  -> {run_name}")

        # Time series data containers
        times = []
        regressive_cells = []
        blastemal_cells = []
        total_cells = []
        dead_cells = []

        # Get all timepoint files
        mat_files = sorted([
            f for f in os.listdir(run_path)
            if re.match(r'output\d+_cells.mat', f)
        ])

        for f in mat_files:
            match = re.search(r'output(\d+)_cells.mat', f)
            if not match:
                continue

            time_index = int(match.group(1))
            t = time_index * time_step

            mat_path = os.path.join(run_path, f)
            try:
                data = scipy.io.loadmat(mat_path)
                cells = data['cells']  # rows = attributes, columns = cells

                cell_types = cells[5, :]      # Row 6 = cell_type
                dead_flags = cells[26, :]     # Row 27 = dead

                regressive = np.sum(cell_types == 0)
                blastemal = np.sum(cell_types == 1)
                total = cells.shape[1]
                dead = np.sum(dead_flags == 1)

                times.append(t)
                regressive_cells.append(regressive)
                blastemal_cells.append(blastemal)
                total_cells.append(total)
                dead_cells.append(dead)

            except Exception as e:
                print(f"Failed reading {mat_path}: {e}")

        # Plot 1: Regressive vs Blastemal
        if times:
            plt.figure(figsize=(8, 5))
            plt.plot(times, regressive_cells, label='Regressive', color='blue')
            plt.plot(times, blastemal_cells, label='Blastemal', color='orange')
            plt.title(f"{run_name} ({cond.upper()}): Regressive vs Blastemal")
            plt.xlabel("Time (min)")
            plt.ylabel("Cell Count")
            plt.legend()
            plt.tight_layout()
            #plt.show()

            # Plot 2: Total vs Dead
            plt.figure(figsize=(8, 5))
            plt.plot(times, total_cells, label='Total', color='green')
            plt.plot(times, dead_cells, label='Dead', color='red', linestyle='--')
            plt.title(f"{run_name} ({cond.upper()}): Total vs Dead")
            plt.xlabel("Time (min)")
            plt.ylabel("Cell Count")
            plt.legend()
            plt.tight_layout()
            plt.savefig(f"{cond}_{run_name}_regressive_vs_blastemal.png", dpi=300)
            plt.close()

