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

## Usages
* ### Last token embedding shortage

```bash
chmod +x Last_Token_{data_name}.sh
./last_token_{data_name}.sh
```

* ### Train and inference
   
```bash

chmod +x {data_name}.sh
./{data_name}.sh
```
