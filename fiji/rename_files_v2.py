#@ File	(label = "Roi input directory", style = "directory") roiFile
#@ File	(label = "Output Directory", style = "directory") outFolder

import os
import shutil

def copy_and_rename_roi(input_folder, output_folder):
    # Ensure output folder exists
    #os.makedirs(output_folder, exist_ok=True)

    # Loop through all files in the input folder
    for filename in os.listdir(input_folder):
        if filename.endswith("00001.roi"):
            # Build full input path
            src = os.path.join(input_folder, filename)

            # Replace suffix
            new_filename = filename.replace("00001.roi", "00002.roi")

            # Build output path
            dst = os.path.join(output_folder, new_filename)

            # Copy file
            shutil.copy2(src, dst)
            print "Copied: {src} to {dst}" 

if __name__ == "__main__":
    # Example usage
    input_folder = roiFile.getAbsolutePath()

    output_folder = outFolder.getAbsolutePath()
    copy_and_rename_roi(input_folder, output_folder)