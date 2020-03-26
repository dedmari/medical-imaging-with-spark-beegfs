# Digital Pathology Imaging Analysis with Artificial Intelligence: Spark and BeeGFS on NetApp E-Series
***********************************************************************************************************
![alt text](https://github.com/dedmari/medical-imaging-with-spark-beegfs/blob/master/architecture_images/patches.png)
***********************************************************************************************************

## Prerequisites
- Python 3.6
- [Spark 2.4.4](https://spark.apache.org/releases/spark-release-2-4-4.html)
- [Tensorflow 1.15](https://www.tensorflow.org/)
- [ASAP](https://computationalpathologygroup.github.io/ASAP/)
- [BeeGFS](https://www.beegfs.io/content/)

## Usage:

1. Creating Patches:
   
       'PYSPARK_PYTHON=python3 spark-submit --master spark:master-node preprocess.py'
       
2. Creating TFRecord from generated Patches:

       'python3 ./utils/build_tfrecord.py'
       
3. Run Training using Inceptionv3 (based on tensorflow slim)
       
       'python3 ./slim/train_image_classifier.py'
       
Configuration and other parameters can be changed in files located in properties directory.
	
***********************************************************************************************************
![alt text](https://github.com/dedmari/medical-imaging-with-spark-beegfs/blob/master/architecture_images/spark_work_flow.png)
***********************************************************************************************************

Contributors: Muneer Ahmad Dedmari and Jürgen Turk

Note: Code is highly influenced by the repo https://github.com/anuragvermaknn/digital-image-analysis-sulli
