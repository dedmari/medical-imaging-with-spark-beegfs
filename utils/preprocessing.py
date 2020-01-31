"""
Preprocess -- Preprocessing and creating Patches (Spark and BeeGFS as distributed file system)
from WSI using Camelyon16 dataset


This module contains functions for the preprocessing phase.
"""

import numpy as np
import pandas as pd

# importing custom packages
import utils.file_utils as wsi_file_utils
import utils.contour_utils as wsi_contour_utils


# Hard-coded labels_df, later replace this with Camelyon16 data labels

def get_hardcoded_labels(total_slides=111):
    slides = list(range(1, total_slides + 1))

    labels_df = pd.DataFrame(slides, columns={'slide_num'})

    labels_df.set_index("slide_num", drop=False, inplace=True)  # use the slide num as index

    return labels_df


def generate_tumor_patches_from_tumor_images(slide_name='tumor_1.tif',
                                             tumor_path='',
                                             mask_path='',
                                             mask_image_resolution_level=5):
    # for tumor_wsi_path, wsi_mask_path in tumor_image_mask_pairs:
    tumor_wsi_path = tumor_path + slide_name
    wsi_mask_path = mask_path + slide_name
    print("tumor_wsi_path", tumor_wsi_path)
    print("wsi_mask_path", wsi_mask_path)
    wsi_mask = wsi_file_utils.read_wsi_normal(wsi_normal_path=wsi_mask_path,
                                              resolution_level=mask_image_resolution_level)

    # TODO: Patches can be saved later. We can create only RDD based on patches here.
    wsi_contour_utils.get_and_save_tumor_patch_samples_for_tumor_images(mask_image=np.array(wsi_mask),
                                                                        mask_image_resolution_level=mask_image_resolution_level,
                                                                        wsi_path=tumor_wsi_path,
                                                                        wsi_mask_path=wsi_mask_path)


def generate_normal_patches_from_normal_images(slide_name='normal_1',
                                               normal_path='',
                                               mask_image_resolution_level=5):
    """
    Args:
      slide_name:
      normal_wsi_path:
      mask_image_resolution_level:

    Returns:

    """
    normal_wsi_path = normal_path + slide_name
    wsi_contour_utils.get_and_save_normal_patch_samples_from_both_images(
        mask_image_resolution_level=mask_image_resolution_level,
        wsi_path=normal_wsi_path,
        wsi_mask_path=None,
        is_tumor_image=False)


def preprocess_tumor_cam(spark, slide_names, tumor_path, mask_path, mask_image_resolution_level):
    """
    Preprocess and create patches from tumor slides.
    Args:
      spark: SparkSession.
      slide_names: List of whole-slide file names to process (without path).
      tumor_path: path of tumor slides
      wsi_mask_path: path of tumor slide masks
      mask_image_resolution_level: resolution level

    """
    spark.sparkContext \
        .parallelize(slide_names) \
        .foreach(lambda slide_name: generate_tumor_patches_from_tumor_images
        (
        slide_name,
        tumor_path,
        mask_path,
        mask_image_resolution_level
    )
                 )


def preprocess_normal_cam(spark, slide_names, normal_path, mask_image_resolution_level):
    """
    Preprocess and create patches from normal slides.
    Args:
      spark: SparkSession.
      slide_names: List of whole-slide file names to process (without path).
      nomral_path: path of normal slides
      mask_image_resolution_level: resolution level

    """
    spark.sparkContext \
        .parallelize(slide_names) \
        .foreach(lambda slide_name: generate_normal_patches_from_normal_images
        (
        slide_name,
        normal_path,
        mask_image_resolution_level
    )
                 )
