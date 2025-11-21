# AI2CCA
## A confidence-based, artificial intelligence pathology model for liver cancer diagnosis

### Installation
1. Clone the repository
   ```
   git clone https://github.com/cyy325yyc/AI2CCA.git
   ```
2. Create the virtual environment via conda
    ```
    conda create -n ai2cca python=3.11
    conda activate ai2cca
    ```
3. Install the dependencies.
    ```
    pip install -r requirements.txt
    cd ai2cca
    ```
   
### Running training and inference
1. Dataset: For the metadata .csv file organization, please refer to *example_metadata.csv*.
2. Patch-level feature extraction can be effected with [Trident](https://github.com/mahmoodlab/TRIDENT).
3. Training:
   ```
   python train_model.py \
   --data_root_dir /path/to/patch_features_h5 \
   --csv_path /path/to/metadata.csv \
   --split_dir /path/to/splits_dir \
   --exp_code ai2cca_train
   ```
   
4. Inference: 
   ```
   python test.py \
   --data_root_dir /path/to/patch_features_h5 \
   --csv_path /path/to/metadata.csv \
   --models_dir ./results/ai2cca_s1 \
   --eps_dir ./results/ai2cca_s1 \
   --results_dir ./results \
   --exp_code ai2cca_test
   ```
   
### Acknowlegements
The project is based on [Trident](https://github.com/mahmoodlab/TRIDENT) and [TITAN](https://github.com/mahmoodlab/TITAN/tree/main). The scripts are partially based on [CLAM](https://github.com/mahmoodlab/CLAM) and [TITAN](https://github.com/mahmoodlab/TITAN/tree/main).
