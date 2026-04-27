# [Revisiting Reconstruction-based Anomaly Detection for Whole Slide Image (IEEE TMI 2026)](https://ieeexplore.ieee.org/abstract/document/11494142)

## Dataset Preparation

We use the **Camelyon16-BMAD** dataset as an example to reproduce the experimental results.

Download the dataset from the following link:

https://drive.google.com/drive/folders/1AC-wWZl_K18CWL2eIxUScoSOoxT4IBuw

The file `Histopathology_AD.zip` corresponds to the **Camelyon16-BMAD** dataset.

After downloading, extract the dataset:

```bash
unzip Histopathology_AD.zip
```

After extraction, the directory structure should look like the following:

```
camelyon16_256/
├── test/
│   ├── good/
│   └── Ungood/
├── train/
│   └── good/
└── valid/
    ├── good/
    └── Ungood/
```

## Configuration

Modify the dataset path and output path in the configuration file.

```
configs/conch/conch_bmad_config_ecr4ad.yaml
```
Update the following fields:
```
data_root: /path/to/dataset
save_path: /path/to/save/results
```

## Foundation Model Weights

Before running training, please update the path of the foundation model weights in:
```
networks/vit_encoder.py
```
For example, when using **CONCH**, modify the following code:
```python
if "conch" == name.lower():
    # your pytorch_model.bin file path here
    model, preprocess = create_model_from_pretrained(
        'conch_ViT-B-16',
        "your pytorch_model.bin file path here"
    )
```
Replace "your pytorch_model.bin file path here" with the actual path to the downloaded CONCH pretrained weights.

## Weights & Biases (Wandb)

Training logs are recorded using **Weights & Biases**.

Before running the training script, add your Wandb API key in `patch_train.py`:

```python
wandb.login(key="your_wandb_key_here")
```

## Training

Run the training script:
```python
python patch_train.py
```

During training, the script will automatically:

- Load the dataset
- Train the model
- Evaluate the model performance
- Log training metrics to Wandb

## Other Datasets
The **Camelyon16** and **GleasonArvaniti** datasets can be found at [https://camelyon17.grand-challenge.org/](https://camelyon17.grand-challenge.org/), and [https://dataverse.harvard.edu/dataset.xhtml?persistentId=doi:10.7910/DVN/OCYCMP](https://dataverse.harvard.edu/dataset.xhtml?persistentId=doi:10.7910/DVN/OCYCMP), respectively.

## Citation

If you find this work useful, please consider citing:

```bibtex

@article{xiao2026revisiting,
  title={Revisiting Reconstruction-based Anomaly Detection for Whole Slide Image},
  author={Xiao, Bin and Wangulu, Collins and van der Kwast, Theodorus and Yousef, George M and Zabihollahy, Fatemeh},
  journal={IEEE Transactions on Medical Imaging},
  year={2026},
  publisher={IEEE}
}
```

## Acknowledgements

This repository is built upon the implementation of [Dinomaly](https://github.com/guojiajeremy/Dinomaly). We sincerely thank the authors for making their code publicly available.
