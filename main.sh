#!/bin/bash
#SBATCH --job-name=write
#SBATCH --partition=long
#SBATCH --output=write.txt
#SBATCH --mem=300G 
#SBATCH --time=1-20:00   


module load anaconda3/4.7.12
source activate py36
#------#SBATCH --nodelist=gpu003
#--- #SBATCH --gres=gpu:0
path_to_data_="../../../Data/Hainich_Subset/"
path_to_output_="../../../Data/Hainich_Subset/"


s#path_to_output_="../Data/forest4D_dlr/"


path_to_data_="../../../Data/lille_tf2/"
path_to_output_="../../../Data/lille_tf2/"


#path_to_data_="../../../Data/semantic3d_tf2/"
#path_to_output_="../../../Data/semantic3d_tf2/"



areas="testing/"
#areas="training/"
#areas="validation/"

#areas="training/,validation/"

n_classes=9
reg_strength=.3
#.3
voxel=.01
label_min=0

#python -u partition/segmentation.py --areas=$areas --n_labels=$n_classes --path_to_data=$path_to_data_ --path_to_output=$path_to_output_ --version="V0" --reg_strength=$reg_strength  --voxel_width=$voxel --gt_index=-1 --rgb_intensity_index=456 --RGB=False

# --rgb_intensity_index=456
# --label_min=$label_min 
#--gt_index=-1 --rgb_intensity_index=345 
#gt 7 -1

python -u partition/write_segments_to_pc.py --areas=$areas --db_test_name=$areas --n_classes=$n_classes --metrics=False --path_to_data=$path_to_data_ --path_to_output=$path_to_output_


