"""
# Copyright 2025 Donnate Bridget Hooft
# Licensed under the Apache License, Version 2.0 (http://www.apache.org/licenses/LICENSE-2.0)
# This script uses Nibabel (BSD license), NumPy (BSD), and standard Python libraries.

Description:
This script augments volumetric medical imaging data (e.g., AMOS or KiTS datasets) by embedding body part regression (BPR) scores 
as an additional image channel alongside the original CT image. The result is a 4D NIfTI file with shape (H, W, D, C), 
where C = 2 (original image and BPR slice score).

Functionality:
- Reads `.nii.gz` medical image volumes and associated JSON files containing cleaned slice-wise BPR scores.
- Combines the original volume with its corresponding slice score map into a multi-channel NIfTI file.
- Saves the augmented image to the specified output directory.

Note:
It is assumed that the input JSON files already contain *cleaned* slice scores. These scores must have been precomputed using 
the BodyPartRegression pipeline (https://github.com/MIC-DKFZ/BodyPartRegression) by Schuegger et al. This script does **not** perform 
any BPR inference or cleaning itself — it relies on external preprocessing to ensure score validity.
"""


import os
import nibabel as nib
import numpy as np
import json

# Define paths for the directories
json_dir = '/vol/ciamspace/datasets/amos_bpr'
imaging_dir = '/vol/ciamspace/datasets/amos22/imagesVa'
output_dir = '/vol/ciamspace/datasets/amos_bpr/imagesVa'

# Ensure the output directory exists
os.makedirs(output_dir, exist_ok=True)

# Loop through each JSON file in the JSON directory
for json_file in sorted(os.listdir(json_dir)):
    # Ensure we're processing only JSON files
    if not json_file.endswith('.json'):
        continue

    # Get the case ID from the JSON file name (e.g., "amos_0001" from "amos_0001.json")
    case_id = os.path.splitext(json_file)[0]

    # Define paths for the imaging and output files
    imaging_path = os.path.join(imaging_dir, f"{case_id}.nii.gz")
    json_path = os.path.join(json_dir, json_file)
    output_path = os.path.join(output_dir, f"{case_id}.nii.gz")

    print(f"Processing case: {case_id}")
    print(f"Imaging path: {imaging_path}")
    print(f"JSON path: {json_path}")
    print(f"Output path: {output_path}")

    # Check if both the imaging and JSON files exist
    if not os.path.exists(imaging_path) or not os.path.exists(json_path):
        print(f"Missing files for {case_id}. Skipping...")
        continue

    try:
        # Step 1: Load the original NIfTI file
        original_nifti = nib.load(imaging_path)
        original_data = original_nifti.get_fdata()
        print(original_data.shape)
        # Step 2: Load the cleaned slice scores from the JSON file
        with open(json_path, 'r') as f:
            json_data = json.load(f)
        cleaned_slice_scores = np.array(json_data["cleaned slice scores"])
        print(cleaned_slice_scores.shape)
        # Step 3: Prepare a new array for combined data
        new_shape = (original_data.shape[0], original_data.shape[1], original_data.shape[2], 2)
        new_data = np.zeros(new_shape)

        # Insert original data into the first channel
        new_data[..., 0] = original_data

        # Create a 3D array for cleaned slice scores
        cleaned_slice_scores_reshaped = np.zeros(original_data.shape)
        print(cleaned_slice_scores_reshaped.shape)
        # Fill the cleaned slice scores into the appropriate z-slice
        for i in range(cleaned_slice_scores.shape[0]):
            cleaned_slice_scores_reshaped[:, :, i] = cleaned_slice_scores[i]

        # Assign the cleaned slice scores to the second channel
        new_data[..., 1] = cleaned_slice_scores_reshaped

        # Step 4: Create and save the new NIfTI image
        new_nifti = nib.Nifti1Image(new_data, original_nifti.affine, header=original_nifti.header)
        nib.save(new_nifti, output_path)
        print(f"Saved new NIfTI file at: {output_path}")

    except Exception as e:
        print(f"Error processing {case_id}: {e}")

# -------------------------Kits23 dataset-----------------------------
# import os
# import nibabel as nib
# import numpy as np
# import json

# # Path to the dataset directory
# dataset_dir = '/u/home/hodo/Documents/kits23-main/dataset'

# # Loop through each case directory
# for case_dir in sorted(os.listdir(dataset_dir)):
#     case_path = os.path.join(dataset_dir, case_dir)

#     # Skip if it's not a directory
#     if not os.path.isdir(case_path):
#         continue

#     print(f"Processing case: {case_dir}")

#     # Define paths for the required files
#     imaging_path = os.path.join(case_path, 'imaging.nii.gz')
#     json_path = os.path.join(case_path, 'imaging.json')
#     output_path = os.path.join(case_path, 'imaging.nii.gz')
#     print(imaging_path, json_path, output_path)
#     # Check if both the imaging and JSON files exist
#     if not os.path.exists(imaging_path) or not os.path.exists(json_path):
#         print(f"Missing files for {case_dir}. Skipping...")
#         continue

#     try:
#         # Step 1: Load the original NIfTI file
#         original_nifti = nib.load(imaging_path)
#         original_data = original_nifti.get_fdata()

#         # Step 2: Load the cleaned slice scores from the JSON file
#         with open(json_path, 'r') as f:
#             json_data = json.load(f)
#         cleaned_slice_scores = np.array(json_data["cleaned slice scores"])

#         # Step 3: Prepare a new array for combined data
#         new_shape = (original_data.shape[0], original_data.shape[1], original_data.shape[2], 2)
#         new_data = np.zeros(new_shape)

#         # Insert original data into the first channel
#         new_data[..., 0] = original_data

#         # Create a 3D array for cleaned slice scores
#         cleaned_slice_scores_reshaped = np.zeros(original_data.shape)

#         # Fill the cleaned slice scores into the appropriate z-slice
#         for i in range(cleaned_slice_scores.shape[0]):
#             cleaned_slice_scores_reshaped[original_data.shape[0] - i - 1, :, :] = cleaned_slice_scores[i]

#         # Assign the cleaned slice scores to the second channel
#         new_data[..., 1] = cleaned_slice_scores_reshaped

#         # Step 4: Create and save the new NIfTI image
#         new_nifti = nib.Nifti1Image(new_data, original_nifti.affine, header=original_nifti.header)
#         nib.save(new_nifti, output_path)
#         print(f"Saved new NIfTI file at: {output_path}")

#     except Exception as e:
#         print(f"Error processing {case_dir}: {e}")

# # 
# import nibabel as nib
# import numpy as np
# import json
# import os

# # Step 1: Load the original NIfTI file
# original_nifti_path = '/u/home/hodo/Documents/kits23-main/dataset/case_00291/imaging.nii.gz'
# original_nifti = nib.load(original_nifti_path)
# original_data = original_nifti.get_fdata()
# print(original_data.shape)

# # Step 2: Load the cleaned slice scores from the JSON file
# json_path = '/u/home/hodo/Documents/kits23-main/dataset/case_00291/imaging.json'
# with open(json_path, 'r') as f:
#     json_data = json.load(f)
# cleaned_slice_scores = np.array(json_data["cleaned slice scores"])
# print(cleaned_slice_scores.shape)
# # Step 3: Prepare a new array
# # The new shape will be (512, 512, 117, 2) for (x, y, z, channel)
# new_shape = (original_data.shape[0], original_data.shape[1], original_data.shape[2], 2)  # (512, 512, 117, 2)
# new_data = np.zeros(new_shape)
# print(new_data.shape)
# # Insert original data into the first channel
# new_data[..., 0] = original_data
# # Create a 3D array for cleaned slice scores
# cleaned_slice_scores_reshaped = np.zeros((original_data.shape[0], original_data.shape[1], original_data.shape[2]))

# # Fill the cleaned slice scores into the appropriate z-slice
# for i in range(cleaned_slice_scores.shape[0]):
#     cleaned_slice_scores_reshaped[original_data.shape[0]-i -1, :, :] = cleaned_slice_scores[i]

# print(cleaned_slice_scores_reshaped.shape)

# # Assign the cleaned slice scores to the second channel
# new_data[..., 1] = cleaned_slice_scores_reshaped
# # Step 4: Create a new NIfTI image
# new_nifti = nib.Nifti1Image(new_data, original_n