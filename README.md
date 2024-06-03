## Dependencies

* Python 3.11
* PyTorch 2.1.2
* cuda 12.1
* torchvision 0.8.0

```bash
> conda env create -f env_ubuntu.yaml
```

## Datasets
The pre-processed datasets can be obtained from [here](https://drive.google.com/drive/folders/13Cg1KYOlzM5C7K8gK8NfC-F3EYxkM3D2).

## Usage
* ### Last toke embedding shortage

```bash
chmod +x get_emb_{data_name}.sh
./get_emb_{data_name}.sh
```

* ### Train and inference
   
```bash

chmod +x {dataset}.sh
./{dataset}.sh
```
