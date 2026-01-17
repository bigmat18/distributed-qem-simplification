import pymeshlab
import os
import argparse
import concurrent.futures
from concurrent.futures import ProcessPoolExecutor

def process_single_file(filename, input_dir, output_dir):
    input_path = os.path.join(input_dir, filename)
    output_path = os.path.join(output_dir, filename)

    try:
        ms = pymeshlab.MeshSet()
        ms.load_new_mesh(input_path)

        try:
            ms.apply_filter('meshing_remove_duplicate_faces')
            ms.apply_filter('meshing_remove_duplicate_vertices')
            ms.apply_filter('meshing_remove_unreferenced_vertices')
            
            ms.apply_filter('meshing_repair_non_manifold_edges', method='Remove Faces')
            ms.apply_filter('meshing_repair_non_manifold_vertices')
            ms.apply_filter('meshing_remove_null_faces')
            
            ms.apply_filter('meshing_remove_unreferenced_vertices')
        except Exception as clean_err:
            print(f"[WARNING] Issue inside worker for {filename}: {clean_err}")

        out_dict = ms.apply_filter('get_topological_measures')
        
        nm_verts = out_dict.get('non_two_manifold_vertices')
        nm_edges = out_dict.get('non_two_manifold_edges')
        
        is_clean = (nm_edges == 0) and (nm_verts == 0)

        if is_clean:
            ms.save_current_mesh(output_path, binary=False)
            return ('SAVED', f"[SAVED] {filename} (ASCII)")
        else:
            return ('DISCARDED', f"[DISCARDED] {filename} (Non-manifold: {nm_edges} edges, {nm_verts} vertices)")

    except Exception as e:
        return ('ERROR', f"[ERROR] Could not process {filename}: {e}")


def process_dataset(input_dir, output_dir, max_workers=None):
    if not os.path.exists(input_dir):
        print(f"Error: Input directory '{input_dir}' does not exist.")
        return

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    files = [f for f in os.listdir(input_dir) if f.lower().endswith('.ply')]
    total_files = len(files)

    if max_workers is None:
        max_workers = os.cpu_count()

    print(f"Processing {total_files} files using {max_workers} parallel workers...")
    
    saved_count = 0
    discarded_count = 0
    errors_count = 0

    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        future_to_file = {
            executor.submit(process_single_file, f, input_dir, output_dir): f 
            for f in files
        }

        for future in concurrent.futures.as_completed(future_to_file):
            status, message = future.result()
            
            print(message)

            if status == 'SAVED':
                saved_count += 1
            elif status == 'DISCARDED':
                discarded_count += 1
            elif status == 'ERROR':
                errors_count += 1

    print("-" * 30)
    print("Processing complete.")
    print(f"Total processed: {total_files}")
    print(f"Total saved:     {saved_count}")
    print(f"Total discarded: {discarded_count}")
    print(f"Total errors:    {errors_count}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--input", required=True, help="Input Folder")
    parser.add_argument("-o", "--output", required=True, help="Output Folder")
    parser.add_argument("-w", "--workers", type=int, default=None, help="Number of parallel workers (default: all CPU cores)")

    args = parser.parse_args()
    
    process_dataset(args.input, args.output, args.workers)
