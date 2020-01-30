# -------------------------------------------------------------
#
# Licensed to the Apache Software Foundation (ASF) under one
# or more contributor license agreements.  See the NOTICE file
# distributed with this work for additional information
# regarding copyright ownership.  The ASF licenses this file
# to you under the Apache License, Version 2.0 (the
# "License"); you may not use this file except in compliance
# with the License.  You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing,
# software distributed under the License is distributed on an
# "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
# KIND, either express or implied.  See the License for the
# specific language governing permissions and limitations
# under the License.
#
# -------------------------------------------------------------

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


def generate_tumor_patches_from_tumor_images(slide_num=1,
                                             tumor_path='/mnt/beegfs/spark/camelyon16_rawdata/training/training/tumor/',
                                             mask_path='/mnt/beegfs/spark/camelyon16_rawdata/training/Ground_Truth_Extracted/Mask/',
                                             mask_image_resolution_level=5):
    # for tumor_wsi_path, wsi_mask_path in tumor_image_mask_pairs:
    tumor_wsi_path = tumor_path + 'tumor_' + str(slide_num).zfill(
        3) + '.tif'
    wsi_mask_path = mask_path + 'tumor_' + str(
        slide_num).zfill(3) + '.tif'
    print("tumor_wsi_path", tumor_wsi_path)
    print("wsi_mask_path", wsi_mask_path)
    wsi_mask = wsi_file_utils.read_wsi_normal(wsi_normal_path=wsi_mask_path,
                                              resolution_level=mask_image_resolution_level)

    # Patches can be saved later. We can create only RDD based on patches here. TODO: by Muneer
    wsi_contour_utils.get_and_save_tumor_patch_samples_for_tumor_images(mask_image=np.array(wsi_mask),
                                                                        mask_image_resolution_level=mask_image_resolution_level,
                                                                        wsi_path=tumor_wsi_path,
                                                                        wsi_mask_path=wsi_mask_path)


def generate_normal_patches_from_normal_images(slide_num=1,
                                               normal_path='/mnt/beegfs/spark/camelyon16_rawdata/training/training/normal/',
                                               mask_image_resolution_level=5):
    """
    Args:
      slide_num:
      normal_wsi_path:
      mask_image_resolution_level:

    Returns:

    """
    normal_wsi_path = normal_path + 'normal_' + str(slide_num).zfill(
        3) + '.tif'
    wsi_contour_utils.get_and_save_normal_patch_samples_from_both_images(
        mask_image_resolution_level=mask_image_resolution_level,
        wsi_path=normal_wsi_path,
        wsi_mask_path=None,
        is_tumor_image=False)


def preprocess_tumor_cam(spark, slide_nums, tumor_path, mask_path, mask_image_resolution_level):
    """
    Preprocess and create patches from tumor slides.
    Args:
      spark: SparkSession.
      slide_nums: List of whole-slide numbers to process.
      tumor_path: path of tumor slides
      wsi_mask_path: path of tumor slide masks
      mask_image_resolution_level: resolution level

    """
    spark.sparkContext \
        .parallelize(slide_nums) \
        .foreach(lambda slide_num: generate_tumor_patches_from_tumor_images
        (
        slide_num,
        tumor_path,
        mask_path,
        mask_image_resolution_level
    )
                 )


def preprocess_normal_cam(spark, slide_nums, normal_path, mask_image_resolution_level):
    """
    Preprocess and create patches from normal slides.
    Args:
      spark: SparkSession.
      slide_nums: List of whole-slide numbers to process.
      nomral_path: path of normal slides
      mask_image_resolution_level: resolution level

    """
    spark.sparkContext \
        .parallelize(slide_nums) \
        .foreach(lambda slide_num: generate_normal_patches_from_normal_images
        (
        slide_num,
        normal_path,
        mask_image_resolution_level
    )
                 )