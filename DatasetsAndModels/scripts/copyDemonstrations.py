import os
import shutil
import argparse

def copy_files(source_dir, target_dir, filenames_file):
    # Read the list of filenames from the text file
    with open(filenames_file, "r") as file_list_file:
        filenames = file_list_file.read().splitlines()

    # Check if the target directory already exists
    if not os.path.exists(target_dir):
        # Create the target directory if it doesn't exist
        os.makedirs(target_dir)

        # Loop through the list of filenames and copy the corresponding .h5 and .json files
        for filename in filenames:
            src_h5_file = os.path.join(source_dir, f"{filename}.h5")
            src_json_file = os.path.join(source_dir, f"{filename}.json")

            if os.path.exists(src_h5_file):
                # Copy the .h5 file
                shutil.copy(src_h5_file, target_dir)
                print(f"Copied {filename}.h5")

                # Check if the corresponding .json file exists
                if os.path.exists(src_json_file):
                    # Copy the .json file
                    shutil.copy(src_json_file, target_dir)
                    print(f"Copied {filename}.json")
                else:
                    print(f"JSON file {filename}.json not found in {source_dir}")
            else:
                print(f"H5 file {filename}.h5 not found in {source_dir}")
    else:
        print(f"Target directory '{target_dir}' already exists. Doing nothing.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Copy selected .h5 and .json files based on a list of filenames.")
    parser.add_argument("-i", "--input-dir", required=True, help="Source directory containing .h5 and .json files")
    parser.add_argument("-o", "--output-dir", required=True, help="Target directory for copied files")
    parser.add_argument("-f", "--filenames-file", required=True, help="Text file containing the list of filenames")
    args = parser.parse_args()

    copy_files(args.input_dir, args.output_dir, args.filenames_file)
