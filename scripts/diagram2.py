import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np
import math

# ==========================================
# 1. CONFIGURAZIONE DATI
# ==========================================

implementations_labels = [
    # "OMP Uniform Grid", 
    # "OMP Octree", 
    # "OMP Uniform Red", 
    # "FF Uniform Red"
    "MPI + OMP",
    "Full MPI"
]

# cores_x = [1, 2, 4, 8, 16, 32]
cores_x = [8, 16, 32, 64, 96, 128, 160, 192, 224]
data = [
    # {
    #     "label": "OMP Uniform Grid",
    #     "x": cores_x,
    #     "raw_times": [
    #         [115445, 115174, 114390], 
    #         [74673.2, 75642.6], 
    #         [53386.9, 52448.4], 
    #         [39244.6, 39267.9], 
    #         [32719.2, 32027.6], 
    #         [29191.1, 29005.9]
    #     ]
    # },
    # {
    #     "label": "OMP Octree",
    #     "x": cores_x,
    #     "raw_times": [
    #         [164737, 163334, 163766], 
    #         [107633, 107710], 
    #         [69533.8, 68850], 
    #         [46813.9, 45738.8], 
    #         [31005.8, 31522.9], 
    #         [28136.1, 29312.7, 29065.9, 28476.9]
    #     ]
    # },
    # {
    #     "label": "OMP Uniform Red",
    #     "x": cores_x,
    #     "raw_times": [
    #         [116327, 114279, 116498], 
    #         [73341, 72502], 
    #         [51433.2, 50748.1], 
    #         [36094.9, 36413.9], 
    #         [27835.8, 28092.3], 
    #         [25444.2, 25474.2]
    #     ]
    # },
    # {
    #     "label": "FF Uniform Red",
    #     "x": cores_x,
    #     "raw_times": [
    #         [114616, 114465, 114124], 
    #         [82894, 81838], 
    #         [56280, 56337], 
    #         [39801.6, 39947.3], 
    #         [30008.7, 29600.9], 
    #         [26852.5, 26826.5]
    #     ]
    # }
    {
        "label": "MPI + OMP",
        "x": cores_x,
        "raw_times": [
            [], # 8
            [], # 16
            [34650, 34710, 34690, 35100, 35320], # 32 
            [28090, 28190, 27460, 28170, 28340], # 64
            [23670, 23790, 23840, 23790, 23850], # 96
            [21870, 21990, 22080, 21770, 21870], # 128
            [20430, 20040, 19880, 19900, 19870], # 160
            [20010, 20010, 19860, 19960, 19890], # 192
            [19750, 19800, 19860, 19790, 19670], # 224
        ]
    },
    {
        "label": "Full MPI",
        "x": cores_x,
        "raw_times": [
            [39480, 39720, 39500], # 8
            [27610, 27690, 27170], # 16
            [28500, 28680, 28360], # 32
            [23710, 23850, 23970], # 64
            [25470, 24790, 24580], # 96
            [22150, 22000, 21400], # 128
            [24150, 25040, 24900], # 160
            [26150, 26000, 25400], # 224
        ]
    }
]

# Configurazione Grafica
x_label = "Number of Cores"
y_label = "Time (ms)"
colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']
markers = ['o', 's', '^', 'D']

# ==========================================
# 2. CALCOLO E PLOTTING
# ==========================================

def create_chart():
    plt.rcParams.update({'font.size': 14})
    fig, ax = plt.subplots(figsize=(10, 8))
    
    all_x = set()
    all_y_values = []

    for i, entry in enumerate(data):
        label = entry["label"]
        x_full = entry["x"]
        raw_times = entry["raw_times"]
        
        valid_x = []
        best_times = [] 
        std_devs = []   
        
        # Iteriamo accoppiando core e liste di tempi corrispondenti
        for core, runs in zip(x_full, raw_times):
            # Se la lista runs esiste, non è None e non è vuota
            if runs is not None and len(runs) > 0:
                best = np.min(runs)
                best_times.append(best)      
                std_devs.append(np.std(runs, ddof=1))
                valid_x.append(core)
                all_y_values.append(best)
        
        all_x.update(valid_x)
            
        # Plotting della linea per l'implementazione corrente
        ax.errorbar(
            valid_x, best_times, yerr=std_devs, label=label, 
            marker=markers[i % len(markers)], color=colors[i % len(colors)],
            linestyle='-', linewidth=2, markersize=8, capsize=5,
            elinewidth=1.5, capthick=1.5
        )
    
    # --- ASSE X ---
    sorted_x = sorted(list(all_x))
    ax.set_xticks(sorted_x)
    ax.get_xaxis().set_major_formatter(ticker.ScalarFormatter())

    # --- ASSE Y ---
    ax.yaxis.set_major_locator(ticker.LogLocator(base=10.0, subs=(1.0, 10.0, 20.0, 50.0), numticks=10))
    y_formatter = ticker.ScalarFormatter()
    y_formatter.set_scientific(False)
    ax.get_yaxis().set_major_formatter(y_formatter)
    ax.yaxis.set_minor_formatter(ticker.NullFormatter())

    ax.set_xlabel(x_label, fontsize=16, fontweight='bold', labelpad=10)
    ax.set_ylabel(y_label, fontsize=16, fontweight='bold', labelpad=10)
    
    ax.tick_params(axis='y', labelsize=12, width=1, length=5) 
    ax.tick_params(axis='x', labelsize=14)
    ax.grid(True, which="major", linestyle='--', alpha=0.5, linewidth=1)
    
    ax.legend(fontsize=12, loc='upper right', frameon=True, shadow=True)

    plt.tight_layout()
    output_filename = 'out/scalability_detailed_2.pdf'
    plt.savefig(output_filename, dpi=300)
    print(f"Grafico salvato come {output_filename}")
    plt.show()


if __name__ == "__main__":
    create_chart()