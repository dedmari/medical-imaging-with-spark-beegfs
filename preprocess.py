"""
Preprocess -- Preprocessing and creating Patches (Spark and BeeGFS as distributed file system) from WSI using Camelyon16 dataset

"""

# System libraries
import os
import glob
import shutil

# Spark libraries
from pyspark.sql import SparkSession

# custom package
from utils.preprocessing import get_hardcoded_labels, \
    preprocess_tumor_cam, preprocess_normal_cam

import utils.build_image_data as build_tfrecord
from properties import disk_storage as disk_storage_props


def main(process_normal_slides=True, process_tumor_slides=False, create_tfrecord=False):
    # Create new SparkSession
    spark = (SparkSession.builder
             .appName("Camelyon 16 -- Preprocessing using Spark, BeeGFS on NetApp E-Series")
             .getOrCreate())

    # Send a copy of `utils` and 'properties' packages to the Spark workers.
    dirname = "utils"
    zipname = dirname + ".zip"
    shutil.make_archive(dirname, 'zip', dirname + "/..", dirname)
    spark.sparkContext.addPyFile(zipname)

    dirname = "properties"
    zipname = dirname + ".zip"
    shutil.make_archive(dirname, 'zip', dirname + "/..", dirname)
    spark.sparkContext.addPyFile(zipname)

    ###### Settings Parameters #######
    tumor_path = disk_storage_props.RAW_TUMOR_DATA_DIR
    normal_path = disk_storage_props.RAW_NORMAL_DATA_DIR
    mask_path = disk_storage_props.RAW_TUMOR_MASK_DIR
    mask_image_resolution_level = 5
    ###### End Settings Parameters #######

    # Retrieving tumor and normal slide file names (without path)
    tumor_slide_files = [os.path.basename(x) for x in glob.glob(mask_path + '*.tif')]
    normal_slide_files = [os.path.basename(x) for x in glob.glob(normal_path + '*.tif')]

    # Process tumor and normal slides
    if process_tumor_slides:
        preprocess_tumor_cam(spark, tumor_slide_files, tumor_path, mask_path, mask_image_resolution_level)
    if process_normal_slides:
        preprocess_normal_cam(spark, normal_slide_files, normal_path, mask_image_resolution_level)

    # Creating TFRecord for training based on patches created before
    if create_tfrecord:
        build_tfrecord.create_tf_record_cam()


if __name__ == '__main__':
    main(process_normal_slides=True, process_tumor_slides=False, create_tfrecord=True)
