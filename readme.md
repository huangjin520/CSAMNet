<table>
<tr>
<td align="left">
  <h1 style="margin: 0;">
    <!-- <em>IEEE TMI 2025</em><br> -->
    CSAMNet: Cross-Scale Attention Modulation for Histopathological Nuclei Segmentation
  </h1>

<p align="left"> 
  <span style="background-color: #f0f0f0; padding: 5px 12px; border-radius: 8px; font-weight: 600; font-size: 16px;"> 
    <img src="https://img.icons8.com/ios-filled/20/000000/code.png" alt="Code Icon" style="vertical-align: middle; margin-right: 5px;"> 
    Developed by <a href="mailto:wangshengqian@whu.edu.cn" style="text-decoration: none; color: inherit;">Shengqian Wang</a> 
    &nbsp;&nbsp;|&nbsp;&nbsp; 
    <img src="https://img.icons8.com/ios-filled/20/000000/design.png" alt="Design Icon" style="vertical-align: middle; margin-right: 5px;"> 
    Designed by <a href="mailto:jinhuang@whu.edu.cn" style="text-decoration: none; color: blue;">Jin Huang</a> 
  </span> 
</p>
</td>

<td align="right">
  <img src="img/CSAMNet_logo.jpg" alt="logo" width="280">
</td>
</tr>
</table>


<p align="center">
  <img src="img/Fig_Net__CSAMNet.jpg" alt="CSAMNet" width="100%">
  <br>
  <rm>Figure 1: CSAMNet Architecture</rm>
</p>

- **📖Title：** CSAMNet: Cross-Scale Attention Modulation for Histopathological Nuclei Segmentation

<!-- - **✨Developed by Wang Shengqian | Designed by Huang Jin** -->

- **👨‍💻Author：** Jin Huang · Shengqian Wang · Mengping Long · Taobo Hu · Zhaoyi Ye · Yueyun Weng · Du Wang · Sheng Liu (*Fellow, IEEE*) · Liye Mei · Cheng Lei

- **📬 Corresponding Authors** Liye Mei · liyemei@whu.edu.cn   | Cheng Lei · leicheng@whu.edu.cn  

- **Link：** [![GitHub](https://img.shields.io/badge/GitHub-CSAMNet-black?logo=github)](https://github.com/huangjin520/CSAMNet) [![Paper](https://img.shields.io/badge/Paper-coming%20soon-lightgrey?logo=readthedocs)]() [![Website](https://img.shields.io/badge/Project-Website-blue?logo=google-chrome)](https://www.lei-whu.com)


**📜Abstract:** <p align="justify"> Accurate nuclei segmentation in histopathological images is a fundamental task in computational pathology, enabling downstream applications such as tumor grading, cellular phenotyping, and morphological analysis. However, it remains challenging due to the complex tissue architecture, diverse nuclear morphology, and blurred boundaries. To address these issues, we propose CSAMNet, a framework with four key innovations. First, we introduce the Cross-Scale Attention Modulation (CSAM) mechanism, which bridges the gap between hierarchical semantics and local details via dual-branch attention pathways. Second, we design the Detail-Context Fusion (DCF) block, which captures high-frequency contextual features. Third, we conduct comprehensive evaluations across four imaging modalities and nine datasets, demonstrating robust performance and effective semantic modulation across scales. Fourth, our framework supports full-resolution inference on whole slide images (WSIs), enabling deployment in large-scale clinical scenarios. In summary, our model achieves state-of-the-art accuracy while reducing model complexity and inference time, making it well-suited for both research and clinical applications. The code is available at [CSAMNet](https://github.com/huangjin520/CSAMNet).

# Introduction
This is an official implementation of [CSAMNet: Cross-Scale Attention Modulation for Histopathological Nuclei Segmentation](). ...



## 🚀 Quick start
### 1️⃣ Installation
Assuming that you have installed PyTorch and TorchVision, if not, please follow the [officiall instruction](https://pytorch.org/) to install them firstly. 
Intall the dependencies using cmd:

``` sh
python -m pip install -r requirements.txt --user -q
```

All experiments use the PyTorch 1.8 framework in a Python 3.10 environment. Other versions of pytorch and Python are not fully tested.
### 📂 Data preparation
We have evaluated segmentation performance on Four nuclei segmentation datasets: 
- [🔬CPM17](https://www.frontiersin.org/journals/bioengineering-and-biotechnology/articles/10.3389/fbioe.2019.00053/full)  
- [🔬Kumar](https://ieeexplore.ieee.org/abstract/document/7872382)  
- [🔬MoNuSeg](https://ieeexplore.ieee.org/abstract/document/8880654)  
- [🔬PUMA](https://academic.oup.com/gigascience/article/doi/10.1093/gigascience/giaf011/8024182) 

Four other modality datasets:
<div align="center">

Table: Summary of Datasets Used for Nuclei Segmentation
<table>
  <thead>
    <tr>
      <th>Dataset</th>
      <th>Organ</th>
      <th># Nuclei</th>
      <th>Image Size</th>
      <th>Images</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>CPM17</td>
      <td>Brain, Head, Neck, Lung</td>
      <td>7,570</td>
      <td>500–600</td>
      <td>32 (WSI)</td>
    </tr>
    <tr>
      <td>Kumar</td>
      <td>Breast, Liver, Kidney, Prostate, Bladder, Colon, Stomach</td>
      <td>21,623</td>
      <td>1000×1000</td>
      <td>30 (WSI)</td>
    </tr>
    <tr>
      <td>MoNuSeg</td>
      <td>Breast, Liver, Kidney, Prostate, Bladder, Colon, Stomach</td>
      <td>28,846</td>
      <td>1000×1000</td>
      <td>44 (WSI)</td>
    </tr>
    <tr>
      <td>PUMA</td>
      <td>Melanoma</td>
      <td>97,429</td>
      <td>1024×1024</td>
      <td>206 (ROI)</td>
    </tr>
  </tbody>
</table>

</div>

- [🎀Dataset B](https://ieeexplore.ieee.org/abstract/document/8003418)
- [🧠Brain Tumor MRI](https://figshare.com/articles/dataset/brain_tumor_dataset/1512427)  
- [📂LIDC-IDRI](https://wiki.cancerimagingarchive.net/display/Public/LIDC-IDRI)  
- [🎀BACH2018](https://iciar2018-challenge.grand-challenge.org/dataset/)  


Dataset tree:
📂 DATASET  
└── 📂 MoNuSeg  
&emsp; ├── 📂 train  
&emsp; │ &emsp; ├── img  
&emsp; │ &emsp; └──  mask  
&emsp; ├── 📂 val  
&emsp; │ &emsp; ├──  img  
&emsp; │ &emsp; └── mask  
&emsp; └── 📂 test  
&emsp; &emsp; ├──img  
&emsp; &emsp; └── mask


### Training
The CSAMNet model can be trained on training set using the following: 

```
python train_CSAMNet.py 
``` 

The parameters of the model have been carefully designed. 
CSAMNet - Hardware: an NVIDIA RTX 3090 GPU and an Intel Core i9-10900X CPU.
<p align="center">
  <img src="img/Fig_PR_ROC.jpg" alt="CSAMNet" width="100%">
  <br>
  <rm>Figure: PR and ROC curves on the MoNuSeg dataset. TPR denotes True Positive Rate; FPR denotes False Positive Rate.</rm>
</p>


## 📊 Evaluation
The CSAMNet model can be evaluated on validation set using the following: 

```
python eval.py 
``` 

<p align="center">
  <img src="img/Fig_WSI_info.jpg" alt="CSAMNet" width="100%">
  <br>
  <rm>Figure: (a) Tumor burden heatmaps on the public BACH2018 dataset; Red contours denote cancer. (b) Predicted nuclei masks on in-house WSIs.</rm>
</p>

<p align="center">
  <img src="img/Fig_FLOPs.jpg" alt="CSAMNet" width="60%">
  <br>
  <rm>Figure: Comparison of FLOPs and parameter counts of CSAMNet SOTA methods on the MoNuSeg dataset.</rm>
</p>

## 📊 Supplementary Material
<p align="center">
  <img src="img/SM_Fig_WSI_pipline.jpg" alt="CSAMNet" width="100%">
  <br>
  <rm>Figure: Workflow of the CSAMNet-based WSI analysis pipeline.</rm>
</p>

<p align="center">
  <img src="img/SM_Fig_heatmaps.jpg" alt="CSAMNet" width="100%">
  <br>
  <rm>Figure: Tumor Burden Heatmaps Generated by CSAMNet from Breast
Cancer WSIs.</rm>
</p>

## 📬 Contact
For any questions or collaborations, please contact [Jin Huang](mailto:jinhuang@whu.edu.cn), [Shengqian Wang](mailto:wangshengqian@whu.edu.cn) or open an issue on GitHub.


<p align="center">
  <img src="img/ours.jpg" alt="CSAMNet" width="50%">
  <br>
</p>

<p align="center">
    <img src="img/Wuhan_university_school_badge.png" alt="Wuhan University Badge" height="50" style="margin-right: 25px;">
    <img src="img/Wuhan_university_name.png" alt="Wuhan University Name" height="50" style="margin-right: 25px;">
    <img src="img/Wuhan_Integrated_Circuits.png" alt="Wuhan Integrated Circuits" height="50">
    <img src="img/School of Robotics.png" alt="School of Robotics" height="50">
</p>




****