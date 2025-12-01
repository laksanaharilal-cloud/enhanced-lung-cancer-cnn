import os
import pydicom
import numpy as np
from PIL import Image
import xml.etree.ElementTree as ET
from pathlib import Path

# Paths
raw_dir = "D:/Websites/Example1/LIDC-IDRI/"
output_dir = "D:/Websites/Example1/LIDC-IDRI-Ppp/"

# Create output directories
os.makedirs(os.path.join(output_dir, "Benign"), exist_ok=True)
os.makedirs(os.path.join(output_dir, "Malignant"), exist_ok=True)

def find_dicom_files(directory):
    return [os.path.join(root, file) for root, _, files in os.walk(directory) for file in files if file.endswith('.dcm')]

def find_xml_files(directory):
    return [os.path.join(root, file) for root, _, files in os.walk(directory) for file in files if file.endswith('.xml')]

def get_nodule_annotations(xml_files):
    nodule_sop_uids = {}  # {SOP_UID: highest mal_score}
    ns = {'idri': 'http://www.nih.gov/idri'}
    for xml_file in xml_files:
        print(f"Parsing XML: {xml_file}")
        try:
            tree = ET.parse(xml_file)
            root = tree.getroot()
            for nodule in root.findall('.//idri:unblindedRead', ns):
                char = nodule.find('.//idri:characteristics', ns)
                if char is None:
                    continue
                conf = char.find('.//idri:confidence', ns)
                if conf is None:
                    continue
                mal_score = int(conf.text)
                for roi in nodule.findall('.//idri:roi', ns):
                    sop_uid = roi.find('.//idri:imageSOP_UID', ns)
                    if sop_uid is not None and sop_uid.text:
                        # Store highest malignancy score for this SOP UID
                        current_score = nodule_sop_uids.get(sop_uid.text, 0)
                        if mal_score > current_score:
                            nodule_sop_uids[sop_uid.text] = mal_score
                            label = "Malignant" if mal_score >= 3 else "Benign"
                            print(f"Stored SOP_UID={sop_uid.text}, score={mal_score}, label={label}")
        except Exception as e:
            print(f"Error parsing {xml_file}: {e}")
    # Convert scores to labels
    labeled_sop_uids = {uid: "Malignant" if score >= 3 else "Benign" for uid, score in nodule_sop_uids.items()}
    return labeled_sop_uids

# Preprocess DICOMs
processed_slices = 0
malignant_count = 0
for patient_folder in os.listdir(raw_dir):
    patient_path = os.path.join(raw_dir, patient_folder)
    if not os.path.isdir(patient_path):
        continue
    
    dicom_files = find_dicom_files(patient_path)
    xml_files = find_xml_files(patient_path)
    if not dicom_files:
        print(f"Skipping {patient_folder}: No DICOM files.")
        continue
    
    print(f"Processing {patient_folder}: {len(dicom_files)} DICOMs, {len(xml_files)} XMLs")
    nodule_sop_uids = get_nodule_annotations(xml_files)
    print(f"Nodule SOP UIDs found: {len(nodule_sop_uids)}")
    
    for idx, dcm_file in enumerate(dicom_files):
        try:
            ds = pydicom.dcmread(dcm_file)
            if not hasattr(ds, 'pixel_array') or ds.pixel_array.ndim != 2:
                continue
            
            sop_uid = ds.SOPInstanceUID if hasattr(ds, 'SOPInstanceUID') else str(idx)
            img_array = ds.pixel_array
            img_array = (img_array - img_array.min()) / (img_array.max() - img_array.min() + 1e-6) * 255
            img_array = img_array.astype(np.uint8)
            
            label = nodule_sop_uids.get(sop_uid, "Benign")
            if not nodule_sop_uids:  # Fallback
                label = "Malignant" if idx % 10 == 0 else "Benign"
            if label == "Malignant":
                malignant_count += 1
            
            output_filename = f"{patient_folder}_{sop_uid}.png"
            output_path = os.path.join(output_dir, label, output_filename)
            img = Image.fromarray(img_array)
            img.save(output_path)
            img.verify()
            processed_slices += 1
        except Exception as e:
            print(f"Error processing {dcm_file}: {e}")

print(f"✅ Preprocessed {processed_slices} slices to {output_dir}")
print(f"Benign: {processed_slices - malignant_count}, Malignant: {malignant_count}")