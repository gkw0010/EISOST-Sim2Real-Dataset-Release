<div align="center">

<samp>

<h2> Domain Adaptive Sim-to-Real Segmentation of Oropharyngeal Organs </h1>

<h4> Guankun Wang, Tian-Ao Ren, Jiewen Lai, Long Bai, and Hongliang Ren </h3>

</samp>   

</div>     

## Dataset: Endoscopic Images generated from SOFA-based oropharynx model with style transfer from phantom (EISOST)
EISOST is a Sim-to-Real oropharyngeal organs segmentation dataset, including 1397 labeled images. The dataset consists of 3 necessary oropharyngeal organs: the uvula, epiglottis, and glottis. Training data (source image) includes 1194 images sampled from the SOFA-based oropharynx model. Test data (test image) contains 203 images captured on a real-world phantom. For the annotations, we provide coarse and fine annotations at the pixel level, including instance-level labels for oropharyngeal organs.

![Image text](https://github.com/gkw0010/EISOST-Sim2Real-Dataset-Release/blob/main/Representative_image.png)

## Image Style-Transfer for Domain Adaption
To reduce the differences between the two datasets, we try to introduce the style-transfer method. With the help of [ArtFlow](https://github.com/pkuanjie/ArtFlow), we convert the appearance of virtual images into real oropharyngeal organs' appearance, thereby enhancing the sense of photo-realistic of virtual data while preserving useful anatomical features for model training. The transfer content and result (transfer image) of the representative image are shown below.

![Image text](https://github.com/gkw0010/EISOST-Sim2Real-Dataset-Release/blob/main/Style-Transfer.png)

## Dataset Download
[[Source Image](https://mycuhk-my.sharepoint.com/:u:/g/personal/1155161502_link_cuhk_edu_hk/EXdFnybwGa5MoqRAgaeExwgBry9yWO4M-iMt08LOKFAhtQ?e=uAPSGq)]
[[Target Image](https://mycuhk-my.sharepoint.com/:u:/g/personal/1155161502_link_cuhk_edu_hk/EV4mm4KVw4pLpDNToYDc9gUBAicfeRgpyWNX0B-pVIBl0w?e=LLNEk2)]
[[Trans Image](https://mycuhk-my.sharepoint.com/:u:/g/personal/1155161502_link_cuhk_edu_hk/ETGsia4hUBFCj3-cZPt6uukBEq0INvuBYz115pVTsj7jJg?e=1Ejg4R)]

## Domain Adaptive Sim-to-Real with IRB-AF

![Image text](https://github.com/gkw0010/EISOST-Sim2Real-Dataset-Release/blob/main/flowchat.png)

To alleviate the rapid degradation of segmentation performance due to large differences between datasets, we introduce our domain adaption segmentation in two aspects. The first is IoU-Ranking Blend which is a compelling dataset blending strategy used for the Sim-to-Real training. Another mothod is image style-transfer. It is used to further reduce the differences between the source domain and the target domain through image style. We integrate the above two methods and propose IRB-AF that aligns the image distributions of different datasets in terms of content and style.

The details of IRB-AF will be presented in our work **Domain Adaptive Sim-to-Real Segmentation of Oropharyngeal Organs**(under review). 

The source code for our baseline model comes from [Transfer Learning Library](https://github.com/thuml/Transfer-Learning-Library)

The source code for image style-transfer comes from [ArtFlow](https://github.com/pkuanjie/ArtFlow)

A simple implementation of image style-transfer is included in this repository. You can find it in the directory `easycode2style-transfer`. A typical usage is

```shell script
# you can change the root of content images and style images in styletransfer.py
# During style-transfer, style images will be randomly selected and transferred to the content images
python easycode2style-transfer/styletransfer.py
```

If you find our code or paper useful, please cite as

```bibtex
@article{wang2023domain,
  title = {Domain adaptive Sim-to-Real segmentation of oropharyngeal organs},
  author = {Wang, Guankun and Ren, Tian-Ao and Lai, Jiewen and Bai, Long and Ren, Hongliang},
  journal = {Medical & Biological Engineering & Computing},
  year={2023},
  publisher={Springer}
}
```

## ICRA 2023 Workshop: Sim-to-Real Segmentation in Robot-assisted Transoral Tracheal Intubation
Related work on Sim-to-Real Segmentation has been accepted by ICRA 2023 Workshop. Our poster won the Best Poster Award and can be accessed in EISOST-Sim2Real-Dataset-Release/ICRA 2023 Workshop Poster/

## Attention
EISOST dataset is free for research purpose only. For any questions about the dataset, please contact: gkwang@link.cuhk.edu.hk.
