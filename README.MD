**UNDER CONSTRUCTION**

# G-SWaN: Graph Self-attention WaveNet for Traffic Forecasting

![System Architecture](/GAWN_sysarch02.png)

This is the official PyTorch implementation of the following paper: [Because Every Sensor Is Unique, so Is Every Pair: Handling Dynamicity in Traffic Forecasting](https://arxiv.org/abs/2302.09956). In IoTDI ’23.

If you are interested in traffic forecasting, check out my collection of traffic forecasting papers: https://github.com/aprbw/traffic_prediction.

# Requirements

python==3.7.4
```
numpy==1.17.2
pandas==1.1.1
scipy==1.3.1
torch==1.5.1
```

# Data

Due to size conern, only the smallest dataset (PeMS-08) is included in this repo. The other three datsets can be found by following the instructions below:

The dataset for METR-LA and PeMS-BAY can be downloaded by following the instruction from the [DCRNN github page](https://github.com/liyaguang/DCRNN/tree/master).

The dataset for PeMS-04 (and also PeMS-08) can be downloaded by following the instruction from the [ASTGCN github page](https://github.com/Davidham3/ASTGCN/tree/master) or the [STSGCN github page](https://github.com/Davidham3/STSGCN).

You can find more traffic datasets (not used in the paper) from my collection: https://github.com/aprbw/traffic_prediction

# Run

Check this file: [run_pems08.txt](run_pems08.txt)
This includes the hyperparameters.
All datasets have the same hyperparameters.
However, don't forget to change these arguments:
```
--adjdata
--data_fn
--device
--enb_output_filepath
--project_name
--run_name
--save
--sweep_name
```

# Cite

Harvard:
```
Arian Prabowo, Wei Shao, Hao Xue, Piotr Koniusz, and Flora D. Salim. 2023.
Because Every Sensor Is Unique, so Is Every Pair: Handling Dynamicity in Traffic Forecasting.
In International Conference on Internet-of-Things Design and Implementation (IoTDI ’23),
May 9–12, 2023, San Antonio, TX, USA. ACM, New York, NY, USA, 20 pages.
https://doi.org/10.1145/3576842.358236
```

BibTeX
```
@inproceedings{prabowo2023GSWaN,
  title={Because Every Sensor Is Unique, so Is Every Pair: Handling Dynamicity in Traffic Forecasting},
  author={Arian Prabowo and Wei Shao and Hao Xue and Piotr Koniusz and Flora D. Salim},
  booktitle={International Conference on Internet-of-Things Design and Implementation (IoTDI ’23)},
  doi={10.1145/3576842.358236},
  url={https://github.com/aprbw/G-SWaN},
  address={San Antonio, TX, USA. ACM, New York, NY, USA},
  year={2023}
}
```
