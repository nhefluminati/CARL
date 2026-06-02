Tools for training Neural Network Classifiers for Likelihood Ratio Estimation (CARL) 
Based on pytorch lightning
Support for ensemble training and various diagnostics (density reweighting, calibration curves, ROC curves, etc.)
Example: 
python CARL/train_CARL_ensemble.py --name my_name --signal my_signal.h5  --backgrounds my_background1.h5  my_background2.h5  --n-ensemble 8 --gpus 0 1 2 3 4 --learning-rate 1e-7  --max-epochs 100  --bootstrap-fraction 0.8 --output-dir carl_ensemble_outputs/
