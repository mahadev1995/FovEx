# FovEx: Model-Agnostic Foveation-based Visual Explanations for Deep Neural Networks

This repository contains the code and example use case for the paper "FovEx: Model-Agnostic Foveation-based Visual Explanations for Deep Neural Networks" submitted in European Conference on Computer Vision (ECCV) 2024.

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