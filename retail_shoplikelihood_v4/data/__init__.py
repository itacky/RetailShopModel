################################
# Data Engine Workflow
################################

#Feature Engineering Steps:
# 1a) generate_embeddings.py - Generates lightfm embeddings for customers and items using LightFM Models stored on GCP [GCP Compute]
# 1b) create_customer_features.py - Creates customer features for segmentation [Datalake Compute]
# 1c) create_customer_labels.py - Creates target labels for model training [Datalake Compute]
# 2a) create_statistic_inputs.py - First step in creating statistical features for customers [Datalake Compute]
# 2b) create_statistic_features.py - Second step in creating statistical features for customers from statistical inputs [Datalake Compute]


# Model Post Processing Steps:
# 1) create_likelihood_labels.py - After Model training and probabilities are generated, this script creates likelihood labels for customers [Datalake Compute]
# 2) merge_final_labels.py - Merges V4 likelihood labels with V3 labels to create final labels for customers [Datalake Compute]