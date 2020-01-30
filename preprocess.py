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
Preprocess -- Preprocessing and creating Patches (Spark and BeeGFS as distributed file system) from WSI using Camelyon16 dataset

"""

# System libraries
import shutil

# Spark libraries
from pyspark.sql import SparkSession

# custom package
from utils.preprocessing import get_hardcoded_labels, \
    preprocess_tumor_cam, preprocess_normal_cam

import utils.build_image_data as build_tfrecord
from properties import disk_storage as disk_storage_props

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


# Hard-coded labels to create patches from Camelyon16 dataset for now.
# TODO: Retrieve all tumor slides (based on tumor labels) and all normal slides and pass that to preprocess functions instead of side numbers.
# Replace harcoded-label function with get_slides(type=normal/tumor), return paths.
tumor_labels_df = get_hardcoded_labels(total_slides=5)
normal_labels_df = get_hardcoded_labels(total_slides=160)

# Process train & val slides
# @Muneer, using hard coded (unless labels are reed properly) training indexes for creating pacthes in test mode. Dataset = Camelyon16


preprocess_tumor_cam(spark, tumor_labels_df.index, tumor_path, mask_path, mask_image_resolution_level)
preprocess_normal_cam(spark, normal_labels_df.index, normal_path, mask_image_resolution_level)

build_tfrecord.create_tf_record_cam()
