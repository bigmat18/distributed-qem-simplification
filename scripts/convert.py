import pymeshlab
import os
import argparse
import sys
import concurrent.futures
from concurrent.futures import ProcessPoolExecutor

def convert_single_entry(entry_name, input_dir, output_dir):
    subdir_path = os.path.join(input_dir, entry_name)

    if not os.path.isdir(subdir_path):
        return None

    obj_path = None
    
    try:
        for f_name in os.listdir(subdir_path):
            if f_name.lower().endswith('.obj'):
                potential_path = os.path.join(subdir_path, f_name)
                if os.path.isfile(potential_path):
                    obj_path = potential_path
                    break 
    except Exception as e:
        return (False, f"[ERROR] Error reading directory {subdir_path}: {e}")

    if obj_path:
        output_filename = f"{entry_name}.ply"
        output_file_path = os.path.join(output_dir, output_filename)

        try:
            ms = pymeshlab.MeshSet()
            ms.load_new_mesh(obj_path)
            ms.save_current_mesh(output_file_path)
            
            return (True, f"{output_file_path}")
            
        except Exception as e:
            return (False, f"[ERROR] Failed to convert {obj_path}: {e}")
    
    return None

def convert_meshes_parallel(input_dir, output_dir, max_workers=None):
    if not os.path.exists(input_dir) or not os.path.isdir(input_dir):
        print(f"Error: Input must be a valid folder: {input_dir}")
        sys.exit(1)

    if not os.path.exists(output_dir) or not os.path.isdir(output_dir):
        print(f"Error: Output must be a valid folder: {output_dir}")
        sys.exit(1)

    entries = os.listdir(input_dir)
    print(f"Scanning {len(entries)} entries in: {input_dir}")
    
    if max_workers is None:
        max_workers = os.cpu_count()
        
    print(f"Starting parallel processing with {max_workers} workers...")

    files_processed = 0

    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        future_to_entry = {
            executor.submit(convert_single_entry, entry, input_dir, output_dir): entry 
            for entry in entries
        }

        for future in concurrent.futures.as_completed(future_to_entry):
            result = future.result()
            
            if result:
                success, message = result
                print(message)
                
                if success:
                    files_processed += 1

    print("-" * 30)
    print(f"Total converted: {files_processed}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="CLI app to convert distributed mesh simplification datasets to PLY (Parallel)")
    parser.add_argument("-i", "--input", required=True, help="Input Folder")
    parser.add_argument("-o", "--output", required=True, help="Output Folder")
    parser.add_argument("-w", "--workers", type=int, default=None, help="Number of parallel workers")

    args = parser.parse_args()

    convert_meshes_parallel(args.input, args.output, args.workers)
