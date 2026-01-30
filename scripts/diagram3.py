import matplotlib.pyplot as plt
import numpy as np

# ==========================================
# 1. CONFIGURAZIONE DATI
# ==========================================

# cores_list = [1, 2, 4, 8, 16, 32]
cores_list = [8, 16, 32, 64, 96, 128, 160, 192]
# component_colors = {
#     "Mesh-Import": "#bdc3c7",   # Grigio chiaro
#     "Pre-Processing":  "#f39c12",   # Arancione
#     "Simplification": "#3498db", # Blu
#     "Post-Processing":      "#e74c3c",   # Rosso
# }

# text_colors = {
#     "Mesh-Import": "black",     # Su grigio chiaro meglio nero
#     "Pre-Processing":  "black",     # Su arancione meglio nero
#     "Simplification": "white", # Su blu meglio bianco
#     "Post-Processing":      "white",     # Su rosso meglio bianco
# }

component_colors = {
    "Recv Waiting": "#bdc3c7",   # Grigio chiaro
    "Processing":  "#f39c12",   # Arancione
    # "Master Work": "#3498db", # Blu
}

text_colors = {
    "Recv Waiting": "black",     # Su grigio chiaro meglio nero
    "Processing":  "black",     # Su arancione meglio nero
    # "Master Work": "white", # Su blu meglio bianco
}

data = {
    "Full MPI": {
        8: [("Recv Waiting", 5205), ("Processing", 5415)],
        16: [("Recv Waiting", 4410), ("Processing", 2750)],
        32: [("Recv Waiting", 5431), ("Processing", 1829)],
        64: [("Recv Waiting", 4525), ("Processing", 915)],
        96: [("Recv Waiting", 5500), ("Processing", 615)],
        128: [("Recv Waiting", 4550), ("Processing", 460)],
        160: [("Recv Waiting", 7250), ("Processing", 360)], 
        192: [("Recv Waiting", 6840), ("Processing", 290)]
    },
    # "MPI+OMP": {
    #    32: [("Recv Waiting", 12280), ("Processing", 22650), ("Master Work", 12)],
    #    64: [("Recv Waiting", 13604), ("Processing", 11692), ("Master Work", 2582)],
    #    96: [("Recv Waiting", 13054), ("Processing", 8171), ("Master Work", 2593)],
    #    128: [("Recv Waiting", 12833), ("Processing", 6496), ("Master Work", 2603)],
    #    160: [("Recv Waiting", 12891), ("Processing", 4567), ("Master Work", 2552)],
    #    192: [("Recv Waiting", 13018), ("Processing", 4426), ("Master Work", 2561)],
    #    224: [("Recv Waiting", 12903), ("Processing", 4410), ("Master Work", 2650)]
    # },
    # "OMP Uniform Grid": {
    #     1:  [("Mesh-Import", 30388), ("Pre-Processing", 18244), ("Simplification", 60711), ("Post-Processing", 6152)],
    #     2:  [("Mesh-Import", 26115), ("Pre-Processing", 10088), ("Simplification", 33793), ("Post-Processing", 5618)],
    #     4:  [("Mesh-Import", 22122), ("Pre-Processing", 6175), ("Simplification", 19092), ("Post-Processing", 6025)], 
    #     8:  [("Mesh-Import", 19524), ("Pre-Processing", 3754), ("Simplification", 13045), ("Post-Processing", 802)],
    #     16: [("Mesh-Import", 17853), ("Pre-Processing", 2120), ("Simplification", 7331), ("Post-Processing", 5564)],
    #     32: [("Mesh-Import", 16981), ("Pre-Processing", 1714), ("Simplification", 6035), ("Post-Processing", 5360)],
    # },
    # "OMP Octree": {
    #     1:  [("Mesh-Import", 30631), ("Pre-Processing", 15811), ("Simplification", 116957), ("Post-Processing", 1039)],
    #     2:  [("Mesh-Import", 25584), ("Pre-Processing", 8590), ("Simplification", 72993), ("Post-Processing", 996)],
    #     4:  [("Mesh-Import", 21757), ("Pre-Processing", 5776), ("Simplification", 41074), ("Post-Processing", 1073)],
    #     8:  [("Mesh-Import", 19322), ("Pre-Processing", 3509), ("Simplification", 22623), ("Post-Processing", 1319)],
    #     16: [("Mesh-Import", 17583), ("Pre-Processing", 2575), ("Simplification", 9583), ("Post-Processing", 2017)],
    #     32: [("Mesh-Import", 17252), ("Pre-Processing", 2297), ("Simplification", 7142), ("Post-Processing", 2548)],
    # },
    # "FF Uniform": {
    #     1:  [("Mesh-Import", 30456), ("Pre-Processing", 19910), ("Simplification", 63537), ("Post-Processing", 996)],
    #     2:  [("Mesh-Import", 25888), ("Pre-Processing", 13974), ("Simplification", 44342), ("Post-Processing", 951)],
    #     4:  [("Mesh-Import", 21913), ("Pre-Processing", 7315), ("Simplification", 26978), ("Post-Processing", 869)],
    #     8:  [("Mesh-Import", 19535), ("Pre-Processing", 4018), ("Simplification", 16025), ("Post-Processing", 833)],
    #     16: [("Mesh-Import", 17862), ("Pre-Processing", 2384), ("Simplification", 9161), ("Post-Processing", 793)],
    #     32: [("Mesh-Import", 16718), ("Pre-Processing", 1903), ("Simplification", 7970), ("Post-Processing", 755)],
    # },
    # "OMP Uniform Red.": {
    #     1:  [("Mesh-Import", 30995), ("Pre-Processing", 19932), ("Simplification", 64070), ("Post-Processing", 989)],
    #     2:  [("Mesh-Import", 26109), ("Pre-Processing", 10687), ("Simplification", 35844), ("Post-Processing", 859)],
    #     4:  [("Mesh-Import", 22896), ("Pre-Processing", 6788), ("Simplification", 21102), ("Post-Processing", 856)],
    #     8:  [("Mesh-Import", 19524), ("Pre-Processing", 3754), ("Simplification", 13045), ("Post-Processing", 805)],
    #     16: [("Mesh-Import", 17517), ("Pre-Processing", 2313), ("Simplification", 7904), ("Post-Processing", 794)],
    #     32: [("Mesh-Import", 16746), ("Pre-Processing", 1937), ("Simplification", 6536), ("Post-Processing", 770)],
    # },
}

def create_final_chart():
    plt.rcParams.update({'font.size': 11})
    fig, ax = plt.subplots(figsize=(18, 10)) # Più largo (18) per dare respiro tra i gruppi

    impl_names = list(data.keys())
    x_indices = np.arange(len(cores_list))  
    bar_width = 0.3 # Ridotta la larghezza per evitare affollamento
    
    num_impl = len(impl_names)
    # Calcolo offset per centrare il blocco di 4 barre sopra il tick del core
    total_block_width = num_impl * bar_width
    start_offset = -total_block_width / 2 + bar_width / 2

    legend_handles = {}
    max_y_value = 0

    # Calcolo il max Y per scalare i testi correttamente
    for impl in data:
        for core in data[impl]:
            max_y_value = max(max_y_value, sum(v for _, v in data[impl][core]))

    for i, impl_name in enumerate(impl_names):
        offset = start_offset + (i * bar_width)
        
        for j, core_count in enumerate(cores_list):
            components = data[impl_name].get(core_count, [])
            total_height = sum([val for _, val in components])
            
            current_bottom = 0
            current_bar_x = x_indices[j] + offset

            for comp_label, comp_value in components:
                color = component_colors.get(comp_label, "#333333")
                txt_color = text_colors.get(comp_label, "white")
                
                bar = ax.bar(current_bar_x, comp_value, bar_width, bottom=current_bottom, 
                             color=color, edgecolor='white', linewidth=0.3)
                
                pct = (comp_value / total_height) * 100
                if pct > 8: # Alzata leggermente la soglia per le percentuali
                    ax.text(current_bar_x, current_bottom + (comp_value / 2), f"{int(pct)}%", 
                            ha='center', va='center', fontsize=10, color=txt_color, fontweight='bold')

                current_bottom += comp_value
                if comp_label not in legend_handles:
                    legend_handles[comp_label] = bar
            
            # Valore totale sopra la barra
            # ax.text(current_bar_x, total_height + (max_y_value * 0.005), f"{int(total_height)}", 
            #         ha='center', va='bottom', fontsize=14, fontweight='bold')

            # Nome implementazione ruotato
            ax.text(current_bar_x, - (max_y_value * 0.01), impl_name, 
                    ha='center', va='top', rotation=90, fontsize=14, color='#444444')

    ax.set_xlabel("Number of Cores", fontsize=13, fontweight='bold', labelpad=40)
    ax.set_ylabel("Execution Time (ms)", fontsize=13, fontweight='bold')
    ax.set_ylim(0, max_y_value * 1.1)
    ax.set_xticks(x_indices)
    ax.set_xticklabels(cores_list, fontsize=16, fontweight='bold')
    ax.tick_params(axis='x', which='major', pad=100) # Aumentato il pad per i numeri dei core
    ax.grid(axis='y', linestyle='--', alpha=0.4)
    
    # Legenda
    ax.legend(legend_handles.values(), legend_handles.keys(), title="Phases", loc='upper right', fontsize=16, title_fontsize=17)

    plt.tight_layout()
    plt.subplots_adjust(bottom=0.28) # Spazio per le label verticali
    
    plt.savefig('out/breakdown_chart_3.pdf', dpi=300)
    plt.show()

if __name__ == "__main__":
    create_final_chart()