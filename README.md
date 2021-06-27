# Non-Exhaustive Learning Using Gaussian Mixture Generative Adversarial Networks

### Authors: Jun Zhuang, Mohammad Al Hasan

### Paper:
 Accepted by ECML-PKDD 2021

### Dataset:
 Three public network intrusion datasets "KDD99", "NSLKDD", "UNSWNB15" and one synthetic dataset

### Getting Started:
#### Prerequisites
 Linux or macOS, CPU, Python 3, keras, numpy, pandas, scikit-learn.

#### Clone this repo
**git clone https://github.com/junzhuang-code/NEGMGAN.git** \
**cd NEGMGAN/negmgan**

#### Install dependencies
For pip users, please type the command: **pip install -r requirements.txt** \
For Conda users, you may create a new Conda environment using: **conda env create -f environment.yml**

#### Directories
##### negmgan:
 1. *gen_Data.py*: Generate qualified dataset from raw data
 1. *preproc.py*: Preprocessing the given datasets
 3. *utils.py*: Utils modules
 4. *model_GMGAN.py*: GM-GAN Models
 5. *model_Imeans.py*: I-means Model
 6. *ablation_ExtractUCs.py*: Ablation study -- Implement GM-GAN to extract UCs on streaming data.
 7. *ablation_DetectUCs.py*: Ablation study -- Implement I-means algorithm to detect the number of new emerging classes on streaming data.
 8. *main.py*: Implement NE-GM-GAN on one streaming batch data.

#### Runs
1. Generate datasets  \
  python [script_name] -DATA_NAME -DATA_DIR \
  e.g., python gen_Data.py KDD99 ../data/

2. Ablation study
  * Extract UCs: python ablation_ExtractUCs.py KDD99 1000
  * Detect UCs: python ablation_DetectUCs.py Syn 2 10 3 200

3. Main function \
  python [script_name] -DATA_NAME -NUM_EPOCHS -Z -WS  \
  e.g., python main.py Syn 1500 3 200
