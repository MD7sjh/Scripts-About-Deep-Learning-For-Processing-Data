import nibabel as nib
import os
from totalsegmentator.python_api import totalsegmentator

if __name__ == "__main__":
    imagesTr_path = r'C:\Users\SongJinHong\DeepLearning\autopet\imagesTr'
    labelsTr_path = r'C:\Users\SongJinHong\DeepLearning\autopet\labelsTr'
    for file in os.listdir(imagesTr_path):
        input_img = nib.load(os.path.join(imagesTr_path,file))
        print('segmenting: ',os.path.join(imagesTr_path,file))
        output_img = totalsegmentator(input_img, ml=True, device="gpu",roi_subset=['spleen','kidney_right','kidney_left','gallbladder','esophagus','liver','stomach','aorta','inferior_vena_cava','pancreas','adrenal_gland_right','adrenal_gland_left','duodenum','urinary_bladder'])
        nib.save(output_img, os.path.join(labelsTr_path,file))
        print('done: ',os.path.join(labelsTr_path,file))