#

import getpass

users = ["spark",
		 "dgxadmin"]

user = getpass.getuser()
print('user: %s' % user)



RAW_DATA_DIR_LIST = {
	"root": "",
	"spark": ""
}

CAMELYON_DIR_LIST = {
	"root": "/mnt/beegfs/spark/camelyon16_rawdata/training/",
	"spark":	"/mnt/beegfs/spark/camelyon16_rawdata/training/"
}
CAMELYON_DIR = CAMELYON_DIR_LIST[user]

RAW_DATA_DIR = CAMELYON_DIR + "training/"
RAW_TUMOR_DATA_DIR = RAW_DATA_DIR + "tumor/"
RAW_NORMAL_DATA_DIR = RAW_DATA_DIR + "normal/"
RAW_TUMOR_MASK_DIR = CAMELYON_DIR + "Ground_Truth_Extracted/Mask/"
# Using for testing purpose only.
# RAW_TUMOR_MASK_DIR = CAMELYON_DIR + "Ground_Truth_Extracted/Test_Mask/"

PREPROCESSING_DATA_DIR = CAMELYON_DIR + "PreProcessing/"
PATCHES_DATA_DIR = PREPROCESSING_DATA_DIR + "Patches/"

PATCHES_RAW_DATA_DIR = PATCHES_DATA_DIR + "raw/"

#PATCHES_TRAIN_DATA_DIR = PATCHES_DATA_DIR + "train/"
PATCHES_TRAIN_DATA_DIR = PATCHES_DATA_DIR + "aug/"
PATCHES_TRAIN_TUMOR_DATA_DIR = PATCHES_TRAIN_DATA_DIR + "tumor/"
PATCHES_TRAIN_NORMAL_DATA_DIR = PATCHES_TRAIN_DATA_DIR + "normal/"

PATCHES_TF_RECORD_DIR = PATCHES_DATA_DIR + "tf_record_directory/"
PATCHES_TRAIN_TF_RECORD_DIR = PATCHES_TF_RECORD_DIR + "train/"

POSTPROCESSING_DATA_DIR = CAMELYON_DIR + "PostProcessing/"

################################### HEATMAP PROPS ##########################
HEATMAP_DATA_DIR = POSTPROCESSING_DATA_DIR + "heatmaps/"

RAW_PATCHES_DIR_TO_GET_HEATMAPS = HEATMAP_DATA_DIR + "raw_input/"
RAW_PATCHES_TF_RECORD_DIR_TO_GET_HEATMAPS = HEATMAP_DATA_DIR + "tf_record_directory/"

WSI_RAW_PATCHES_PARENT_DIR_TO_GET_HEATMAPS = RAW_PATCHES_DIR_TO_GET_HEATMAPS + "WSI_NAME/"
WSI_RAW_PATCHES_DIR_TO_GET_HEATMAPS = RAW_PATCHES_DIR_TO_GET_HEATMAPS + "WSI_NAME/" + "patches/"
WSI_RAW_PATCHES_COUNT_FILE_TO_GET_HEATMAPS = RAW_PATCHES_DIR_TO_GET_HEATMAPS + "WSI_NAME/" + "patches_count.txt"
WSI_RAW_PATCHES_TF_RECORD_DIR_TO_GET_HEATMAPS = RAW_PATCHES_DIR_TO_GET_HEATMAPS + "WSI_NAME/" + "tf_record_directory/"


HEATMAP_OUTPUT_DATA_DIR = HEATMAP_DATA_DIR + "output/"
WSI_HEATMAP_OUPUT_FILE = HEATMAP_OUTPUT_DATA_DIR + "WSI_NAME_heatmap.PNG"
WSI_HEATMAP_WITH_ACTUAL_MASK_OUPUT_FILE = HEATMAP_OUTPUT_DATA_DIR + "WSI_NAME_heatmap_with_actual_mask.PNG"
WSI_HEATMAP_CLEANED_FILE_1 = HEATMAP_OUTPUT_DATA_DIR + "WSI_NAME_heatmap_cleaned_1.PNG"
WSI_HEATMAP_CLEANED_FILE_2 = HEATMAP_OUTPUT_DATA_DIR + "WSI_NAME_heatmap_cleaned_2.PNG"
WSI_HEATMAP_CLEANED_FILE_WITH_ACTUAL_MASK = HEATMAP_OUTPUT_DATA_DIR + "WSI_NAME_heatmap_cleaned_with_actual_mask.PNG"
WSI_HEATMAP_WITH_CLEANED_HEATMAP_WITH_ACTUAL_MASK = HEATMAP_OUTPUT_DATA_DIR + "WSI_NAME_heatmap_with_cleaned_heatmap_with_actual_mask.PNG"
WSI_ACTUAL_MASK_OUPUT_FILE = HEATMAP_OUTPUT_DATA_DIR + "WSI_NAME_actual_mask.PNG"

########################## REVIEW DIRS ############################

HEATMAP_REVIEW_DIR = HEATMAP_DATA_DIR + "review/"
REVIEW_DIR_FOR_BBOX_SELECTED_TO_GET_INPUT_FOR_HEATMAP = HEATMAP_REVIEW_DIR + "bbox_chosen/"
WSI_REVIEW_FILE_FOR_BBOX_SELECTED_TO_GET_INPUT_FOR_HEATMAP = REVIEW_DIR_FOR_BBOX_SELECTED_TO_GET_INPUT_FOR_HEATMAP + "WSI_NAME.PNG"
WSI_REVIEW_FILE_FOR_BBOX_ACCEPTED_TO_GET_INPUT_FOR_HEATMAP = REVIEW_DIR_FOR_BBOX_SELECTED_TO_GET_INPUT_FOR_HEATMAP + "WSI_NAME_with_actual_samples_for_heatmaps.PNG"

REVIEW_DIR_FOR_PREDICTIONS = HEATMAP_REVIEW_DIR + "predictions/"
WSI_REVIEW_FILE_FOR_PREDICTIONS = REVIEW_DIR_FOR_PREDICTIONS + "WSI_NAME.txt"

LOGS_DIR = CAMELYON_DIR + "logs/"
#TRAIN_LOGS_DIR = LOGS_DIR + "train_logs"
TRAIN_LOGS_DIR = LOGS_DIR + "retrain_logs_dr_8_epoc" #"train_logs_dr_30_epoc"
EVAL_LOGS_DIR = LOGS_DIR + "eval_dir/"

# HOME = "/mnt/ai/uni_warwick/repos/digital-image-analysis-sulli/utils/wholeslideimages/"
# DATA_DIR = HOME + "data/"
# LYMPH_DATA_DIR = DATA_DIR + "lymph/"
# MNIST_DATA_DIR = DATA_DIR + "mnist/"
# TRAIN_DIR = DATA_DIR + "train_directory/"

#LOGS_DIR = DATA_DIR + "logs/"



# DIR_FOR_SAVING_NON_TUMOR_PATCHES = TRAIN_DIR + "Normal/"

# Label for Tumor images is 1

labels_file = "/mnt/beegfs/spark//medical-imaging-with-spark-beegfs/properties/" + "labels_file.txt"