# Endoscopic Images generated from SOFA-based oropharynx model with style transfer from phantom (EISOST)
EISOST is a Sim-to-Real oropharyngeal organs segmentation dataset, including 1397 labeled images. The dataset consists of 3 necessary oropharyngeal organs: the uvula, epiglottis, and glottis. Training data (source image) includes 1194 images sampled from the SOFA-based oropharynx model. Test data (test image) contains 203 images captured on a real-world phantom. For the annotations, we provide coarse and fine annotations at the pixel level, including instance-level labels for oropharyngeal organs.

![Image text](https://github.com/gkw0010/EISOST-Sim2Real-Dataset-Release/blob/main/Representative_image.png)

# Image Style-Transfer for Domain Adaption
To reduce the differences between the two datasets, we try to introduce the style-transfer method. With the help of [ArtFlow](https://github.com/pkuanjie/ArtFlow), we convert the appearance of virtual images into real oropharyngeal organs' appearance, thereby enhancing the sense of photo-realistic of virtual data while preserving useful anatomical features for model training. The transfer content and result (transfer image) of the representative image are shown below.

![Image text](https://github.com/gkw0010/EISOST-Sim2Real-Dataset-Release/blob/main/Style-Transfer.png)

# Download
[[source_img](https://drive.google.com/file/d/1uxFKOOY1Nx-4JfxLQzgSaItIzJA-ULt-/view?usp=share_link)]
[[target_img](https://drive.google.com/file/d/1ZI9vwpyDGuzp0poWIfUKoatWRAOvCoUl/view?usp=share_link)]
[[trans_img](https://drive.google.com/file/d/1ZPi29nl1sgoKsoUO6_ESrMdp8o6GYnHs/view?usp=share_link)]

# Domain Adaptive Sim-to-Real with IRB-AF

![Image text](https://github.com/gkw0010/EISOST-Sim2Real-Dataset-Release/blob/main/flowchat.png)

To alleviate the rapid degradation of segmentation performance due to large differences between datasets, we introduce our domain adaption segmentation in two aspects. The first is IoU-Ranking Blend which is a compelling dataset blending strategy used for the Sim-to-Real training. Another mothod is image style-transfer. It is used to further reduce the differences between the source domain and the target domain through image style. We integrate the above two methods and propose IRB-AF that aligns the image distributions of different datasets in terms of content and style.


 # Attention
EISOST dataset is free for research purpose only. For any questions about the dataset, please contact: gkwang@link.cuhk.edu.hk.
