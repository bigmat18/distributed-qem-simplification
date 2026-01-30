import matplotlib.pyplot as plt
import numpy as np


mesh_names = ["Bunny (68K)", "Armadillo (300K)", "Dragon (7.5M)", "Lucy (28M)"]
colors = [
    '#2E86AB',  # Deep sky blue (Sequential)
    '#E74C3C',  # Strong red (MeshLab)
    '#27AE60',  # Vibrant green (OMP Uniform Grid)
    '#F39C12',  # Bright orange (OMP Octree)
    '#9B59B6',  # Vivid purple (OMP Uniform Reduction)
    '#3498DB',  # Electric blue (FF Uniform Reduction)
    '#F1C40F',  # Golden yellow (MPI+OMP 7 Worker (32 Th)) - high contrast with others
    '#E91E63',  # Hot pink (Full MPI (128 MPI Procs))
]

implementations = [
    "Sequential", 
    "MeshLab", 
    "OMP Uniform Grid", 
    "OMP Octree", 
    "OMP Uniform Red", 
    "FF Uniform Red",
    "MPI+OMP",
    "Full MPI"
]

raw_data = [
    # ---------------------------------------------------------
    # 1. Bunny (68K)
    # ---------------------------------------------------------
    [
        [347.964, 350.665, 349.529],    # Sequential
        [381, 368, 361, 417, 390],      # MeshLab
        [364.218, 365.091],             # OMP Uniform Grid
        [247.645, 229.113],             # OMP Octree
        [373.049, 392.903],             # OMP Uniform Reduction
        [383.677, 386.213],             # FF Uniform Reduction
        [380, 377, 381],                # MPI + OMP
        [510, 530, 504],                # Full MPI
    ],
    # ---------------------------------------------------------
    # 2. Armadillo (300K)
    # ---------------------------------------------------------
    [
        [1972.17, 1932.62, 1927.41],    # Sequential
        [2154, 2157, 2051, 2188, 2096], # MeshLab
        [728.833, 728.904],             # OMP Uniform Grid
        [496.275, 493.931],             # OMP Octree
        [470.846, 468.747],             # OMP Uniform Reduction
        [504.312, 504.531],             # FF Uniform Reduction
        [330, 335, 340, 320],           # MPI + OMP
        [480, 470, 468],                # Full MPI
    ],
    # ---------------------------------------------------------
    # 3. Dragon (7.5M)
    # ---------------------------------------------------------
    [
        [38962.1, 38079.8, 38162.5],      # Sequential
        [48451, 48355, 48728],            # MeshLab
        [7623.24, 7672.4],                # OMP Uniform Grid
        [6826.58, 7115.15, 6893.88],      # OMP Octree
        [6094.21, 6063.6],                # OMP Uniform Reduction
        [6763.16, 6922.37],               # FF Uniform Reduction
        [4400, 4480, 4415, 4390],         # MPI+OMP
        [5305, 5048, 4850],               # Full+MPI
    ],
    # ---------------------------------------------------------
    # 4. Lucy (28M)
    # ---------------------------------------------------------
    [
        [184339, 182561, 180496],               # Sequential
        [187449, 177208, 176389],               # MeshLab
        [29686.1, 30500.9],                     # OMP Uniform Grid
        [28631.1, 29807.7, 29560.9, 28971.9],   # OMP Octree
        [25969.2, 25939.2],                     # OMP Uniform Reduction
        [27347.5, 27321.5],                     # FF Uniform Reduction
        [17220, 18590, 18100, 17700],           # MPI+OMP
        [21710, 21850, 21900],                  # Full+MPI
    ]
]


def plot_single_mesh(mesh_id):
    if mesh_id < 0 or mesh_id >= len(mesh_names):
        print(f"Errore: mesh_id deve essere tra 0 e {len(mesh_names)-1}")
        return

    selected_mesh_name = mesh_names[mesh_id]
    selected_data = raw_data[mesh_id]

    means = []
    stds = []
    for impl_runs in selected_data:
        means.append(np.mean(impl_runs))
        stds.append(np.std(impl_runs, ddof=1))

    fig, ax = plt.subplots(figsize=(10, 6))
    
    x_pos = np.arange(len(implementations))
    
    bars = ax.bar(
        x_pos, 
        means, 
        yerr=stds, 
        align='center', 
        alpha=0.9, 
        color=colors, 
        capsize=5,
        edgecolor='white',
        linewidth=0.5
    )
    
    ax.set_title(f"Performance Analysis: {selected_mesh_name}", fontsize=16, fontweight='bold', pad=20)
    ax.set_ylabel('Computation Time (ms)', fontsize=16, fontweight='bold')
    ax.tick_params(axis='y', labelsize=16, width=2, length=6)
    ax.set_xticks(x_pos)
    ax.set_xticklabels(implementations, rotation=45, ha='right', fontsize=16)
    
    for bar in bars:
        height = bar.get_height()
        ax.text(
            bar.get_x() + bar.get_width() / 2.0, 
            height * 1.01,
            f'{int(height)}', 
            ha='center', 
            va='bottom', 
            fontsize=12, 
            fontweight='bold'
        )

    ax.grid(axis='y', linestyle='--', alpha=0.3)
    plt.tight_layout()
    
    filename_clean = selected_mesh_name.split(' ')[0].lower()
    output_filename = f'chart_{filename_clean}.pdf'
    plt.savefig("out/"+output_filename, dpi=500)
    print(f"Grafico generato per {selected_mesh_name}: salvato come '{output_filename}'")
    plt.show()


if __name__ == "__main__":
    print("Seleziona la mesh da visualizzare:")
    for i, name in enumerate(mesh_names):
        print(f"{i}: {name}")
    
    try:
        user_input = int(input("\nInserisci l'ID della mesh (0-3): "))
        plot_single_mesh(user_input)
    except ValueError:
        print("Per favore inserisci un numero valido.")
