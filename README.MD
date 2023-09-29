# G-SWaN: Graph Self-attention WaveNet for Traffic Forecasting

![System Architecture](/GAWN_sysarch02.png)

# Because Every Sensor Is Unique, so Is Every Pair: Handling Dynamicity in Traffic Forecasting

This is the official PyTorch implementation of the following paper: [**Because Every Sensor Is Unique, so Is Every Pair: Handling Dynamicity in Traffic Forecasting**]([https://arxiv.org/abs/2302.09956](https://dl.acm.org/doi/10.1145/3576842.3582362)). In IoTDI ’23. [[slides](/20230510_CPS-IoT_IoTDI_G-SWaN.pdf)] [[ArXiv](https://arxiv.org/abs/2302.09956)] [[Talk](https://youtu.be/rCBQ50ZXI0Y)].

If you are interested in traffic forecasting, check out my collection of traffic forecasting papers: https://github.com/aprbw/traffic_prediction . Check also our next work where we use contrastive learning to find the node embedding of new roads https://github.com/cruiseresearchgroup/forecasting-on-new-roads .

# Abstract

Traffic forecasting is a critical task to extract values from cyber-physical infrastructures, which is the backbone of smart transportation. However owing to external contexts, the dynamics at each sensor are unique. For example, the afternoon peaks at sensors near schools are more likely to occur earlier than those near residential areas. In this paper, we first analyze real-world traffic data to show that each sensor has a unique dynamic. Further analysis also shows that each pair of sensors also has a unique dynamic. Then, we explore how node embedding learns the unique dynamics at every sensor location. Next, we propose a novel module called Spatial Graph Transformers (SGT) where we use node embedding to leverage the self-attention mechanism to ensure that the information flow between two sensors is adaptive with respect to the unique dynamic of each pair. Finally, we present Graph Self-attention WaveNet (G-SWaN) to address the complex, non-linear spatiotemporal traffic dynamics. Through empirical experiments on four real-world, open datasets, we show that the proposed method achieves superior performance on both traffic speed and flow forecasting.

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

Check this file: [run_pems08.txt](run_pems08.txt).
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
https://doi.org/10.1145/3576842.3582362
```

BibTeX
```
@inproceedings{prabowo2023GSWaN,
author = {Prabowo, Arian and Shao, Wei and Xue, Hao and Koniusz, Piotr and Salim, Flora D.},
title = {Because Every Sensor Is Unique, so Is Every Pair: Handling Dynamicity in Traffic Forecasting},
year = {2023},
isbn = {9798400700378},
publisher = {Association for Computing Machinery},
address = {New York, NY, USA},
url = {https://doi.org/10.1145/3576842.3582362},
doi = {10.1145/3576842.3582362},
abstract = {Traffic forecasting is a critical task to extract values from cyber-physical infrastructures, which is the backbone of smart transportation. However owing to external contexts, the dynamics at each sensor are unique. For example, the afternoon peaks at sensors near schools are more likely to occur earlier than those near residential areas. In this paper, we first analyze real-world traffic data to show that each sensor has a unique dynamic. Further analysis also shows that each pair of sensors also has a unique dynamic. Then, we explore how node embedding learns the unique dynamics at every sensor location. Next, we propose a novel module called Spatial Graph Transformers (SGT) where we use node embedding to leverage the self-attention mechanism to ensure that the information flow between two sensors is adaptive with respect to the unique dynamic of each pair. Finally, we present Graph Self-attention WaveNet (G-SWaN) to address the complex, non-linear spatiotemporal traffic dynamics. Through empirical experiments on four real-world, open datasets, we show that the proposed method achieves superior performance on both traffic speed and flow forecasting. Code is available at: https://github.com/aprbw/G-SWaN},
booktitle = {Proceedings of the 8th ACM/IEEE Conference on Internet of Things Design and Implementation},
pages = {93–104},
numpages = {12},
keywords = {sensor networks, cyber-physical systems, intelligent transport systems, spatio-temporal},
location = {San Antonio, TX, USA},
series = {IoTDI '23}
}
```

# Animated version of Figure 2(b) and (c)

Fundamental diagrams shows the dynamics between flow (density) in the x-axis and speed (velocity) in the y-axis. Figure 2(b) shows the real-world data, while Figure 2(c) shows that every sensor has a unique fundamental diagram, suggesting a unique underlying dynamics. The PeMS-D4 dataset is detailed in section 4.1.

![fund diag node 1](fund_diag_node1.gif) Fundamental diagram of a sensor in PeMS-D4 dataset.

![fund_diags_18days](fund_diags_18days_c.gif) Fundamental diagrams of 25 selected sensors in PeMS-D4 dataset showing great diversity.
