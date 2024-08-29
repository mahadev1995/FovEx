# FovEx: Human-inspired Explanations for Vision Transformers and Convolutional Neural Networks

This repository contains the code and example use case for the paper ["FovEx: Human-inspired Explanations for Vision Transformers and Convolutional Neural Networks"](https://arxiv.org/abs/2408.02123v1).

![FovEx](https://github.com/user-attachments/assets/50432df7-cc16-44e9-9fc4-faae5888e620)

Description
----------------------
Here we describe the structure of the contents of the code folder. We provide in folder "models" codes for Vision Transformer model used in the main experiments of the paper. The "images" folder contains some example images from ImageNet-1K validation set. We provide the main code for FovEx in FovEx.py file. Moreover, the explanation.ipynb file contains example use of the FovEx code showcasing the explanation generation process.

    images :                folder containing  example images from ImageNet-1K
    models :                folder containing the code for Vision Transformer
    FovEx.py :              main implementation for FovEx
    classIdx.py :           file containing class id to human readable label mapping 
    example.ipynb :         example use of FovEx code

Generating Explanation
----------------------
FovEx can be used to generate explanation for any pretrained image classification model in the following way:
``` python
from FovEx import FovExWrapper

FovEx = FovExWrapper(downstream_model=model,
                     criterion=criterion,
                     target_function=target_function,
                     image_size=image_size,
                     foveation_sigma=foveation_sigma,
                     blur_filter_size=blur_filter_size,
                     blur_sigma=blur_sigma,
                     forgetting=forgetting,
                     foveation_aggregation=1,
                     heatmap_sigma=heatmap_sigma,
                     heatmap_forgetting=heatmap_forgetting,
                     device='cuda'
                    )

explantion, fixation_points,_, _ = FovEx.generate_explanation(images,                                                                                
                                                              pred_labels, 
                                                              scanpath_length, 
                                                              optimization_steps, 
                                                              lr, 
                                                              random_restart,
                                                              normalize_heatmap
                                                              )
```
Results
----------------------
#### Qualitative Evaluation 
###### ViT-b/16
![vit_viz](https://github.com/user-attachments/assets/c099f685-5e41-4031-a5b6-8d33e29dab63)
###### ResNet-50
![resViz](https://github.com/user-attachments/assets/0669c886-f980-4fac-8c13-2df67951e451)

#### Quantitative Evaluation 
###### ViT-b/16
| Eval. Name | FovEx | gradCAM | GAE | Cls. Emb. | Mean. Pert. | RISE | randomCAM |
|:----------:|:-----:|:-------:|:---:|:---------:|:----------:|:----:|:---------:|
|     Avg. % drop (↓)           |  **13.970**    |    40.057     | 86.207    |     34.862    |      29.753     |   15.763   |     80.714      |
|     Avg. % increase (↑)       |   **30.389**   |     11.469    |  0.799    |     13.329    |      20.549     |   22.189   |      1.789     |
|      Delete (↓)               |   0.240        |     0.157     |  0.172    |    **0.155**  |       0.200     |   0.158   |       0.395    |
|     Insert (↑)                |    **0.840**   |    0.818      |  0.806    |     0.817     |       0.674     |   0.782   |        0.682   |
|      EBPG (↑)                 |    **47.705**  |    41.667     |   39.812  |     39.350    |       40.646    |   42.633   |        35.708   |

###### ResNet-50
| Eval. Name | FovEx | gradCAM | gradCAM++  | Mean. Pert. | RISE | randomCAM |
|:----------:|:-----:|:-------:|:---:|:---------:|:----------:|:----:|
|     Avg. % drop (↓)           |  **11.780**    | 21.718    |     19.863    |      85.973     |   11.885   |     61.317      |
|     Avg. % increase (↑)       |   **61.849**   |  43.669   |     45.069    |      4.700      |   55.489   |     16.729     |
|      Delete (↓)               |   0.151        |  0.108    |     0.113     |      **0.082**  |   0.100   |       0.212    |
|     Insert (↑)                |    **0.374**   |  0.368    |     0.361     |       0.280     |   0.372   |        0.287   |
|      EBPG (↑)                 |    46.977      |**48.658** |     47.412    |       42.725    |   43.312   |        38.118   |
