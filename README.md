# Breast cancer detection using deep learning - An ensemble approach

## Overview
This project explores the use of convolutional neural networks (CNNs) for automated breast cancer detection in mammographic medical images. Using the CBIS-DDSM and Mini-MIAS datasets, I implement, train, and evaluate various transfer learning approaches with pre-trained ResNet-50, DenseNet-121, and Inception-v3 models. The research investigates both individual model performance and ensemble methods to improve classification accuracy and reliability in medical diagnosis applications.

## Objectives
- *Explore and preprocess datasets for mammographic*
- *Implement transfer learning with multiple CNN architectures (Resnet50, DenseNet121, InceptionV3)*
- *Develop progressive fine-tuning strategies with differentiated learning rates*
- *Assess individual model performance using comprehensive evaluation metrics*
- *Design and evaluate ensemble approaches (soft voting, hard voting, weighted voting)*
- *Compare ensemble methods against individual models for clinical decision support*
- *Validate generalization capability across different datasets and imaging protocols*

## Datasets
- **CBIS-DDSM**: https://www.cancerimagingarchive.net/collection/cbis-ddsm/
- **Mini-MIAS**: http://peipa.essex.ac.uk/info/mias.html

## Methodology
**Data Preprocessing:**
- *Image standardization*: Conversion of DICOM and PGM formats to unified RGB representation
- *Normalization*: Application of ImageNet statistics for transfer learning compatibility
- *Data augmentation*: Random transformations including rotation, flipping, translation, and color jittering
- *Stratified splitting*: Balanced train/validation/test splits maintaining class distributions
- *Class balancing*: Implementation of weighted sampling and loss functions to address dataset imbalance

**Training strategy:**
- *Base models*: Pre-trained ResNet-50, DenseNet-121, and Inception-v3 with ImageNet weights
- *Transfer learning approach*: Training classifier head and backbone with differentiated learning rates
- *Fine-tuning depths*: Systematic evaluation of layer unfreezing (last 1-2 blocks)
- *Optimizer comparison*: Adam, AdamW, and SGD with differentiated learning rates
- *Regularization techniques*: Dropout, weight decay, gradient clipping, and early stopping
- *Learning rate scheduling*: ReduceLROnPlateau with validation F1-score monitoring

**Ensemble Methods:**
- *Soft voting*: Probability averaging across model predictions
- *Hard voting*: Majority class voting with tie-breaking mechanisms
- *Weighted soft voting*: Performance-based weighting of model contributions

**Evaluation Framework:**
- **Primary metrics**: F1-score, ROC-AUC, accuracy, precision, recall, specificity
- **Cross-validation**: Stratified validation with consistent random seeds

## Results
- **Transfer Learning**: Classifier and backbone training improves model performance significantly compared to models with ImageNet weights
- **Ensemble Advantage**: 3-5% improvement in recall over individual models
- **Resolution Impact**: 512×512 optimal balance of performance vs. computational cost
- **Domain Shift Challenge**: Significant performance drop on external dataset highlights generalization limits

## Proposed Improvements
- **Patch-based Classification**: Process localized regions for enhanced feature extraction
- **Multi-modal Integration**: Combine imaging with patient metadata (age, family history)
- **Advanced Ensemble Methods**: Stacking, attention-based fusion, meta-learning approaches
- **Interpretability Enhancement**: Gradient-CAM, attention maps for clinical transparency
- **Dataset Diversification**: Include normal cases and broader pathology representation

