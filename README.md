# Neural Science Final Report 
## How to use

### how to train a sample model
```bash
python autoencoder/run.py --do train --model_dir result
```
### how to analyze a trained sample model based on a PCA
```bash
python autoencoder/run.py --do analysis --model_dir result
```  

# Requirements
```
pandas==0.20.3
numpy==1.14.2
scikit-learn==0.19.1
matplotlib==2.2.2
```