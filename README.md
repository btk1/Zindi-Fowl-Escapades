# Zindi-Fowl-Escapades
4th Place Solution for Zindi Fowl Escapades Competition

## References:
Thank you to Johnowhitaker https://zindi.africa/users/Johnowhitaker for the starter notebook:
https://colab.research.google.com/drive/1SdrwQtT16FAGmfssJi5riMjDOQU7IvWE

Mogwai's notebook for implementing mixup with fastai v1:
https://github.com/mogwai/fastai_audio/blob/master/tutorials/03_Environmental_Sound_Classification.ipynb

## Solution:
My solution uses the fastai library (https://docs.fast.ai/). I used 3 different architectures densenet161, densenet201, and resnet34. In each case, I used progressive resizing, starting my training with 224x224 images and then training on 512x512. Then, my final submission was a blend of 4 models (2 densenet161, 1 densenet 201 and 1 resnet34). 

The LBs for the individual models: 
Densenet 161 - 1.155
Densenet 161 - 1.237
Densenet 201 - 1.46
Resnet-34    - 1.255

I had one submission left, and I decided to blend the four models .35, .2, .2, .25, respectively. To my surprise, it scored quite well @ 1.006.

Thank you to Zindi for hosting and everyone for their discussions. I hope that this code and explanation is helpful. If you have any questions, please ask and I will try my best to answer.

## To review for improving the models:

I would like to have tried Kfold cross-validation of my models, but I did not have the time. I also think that adjusting the parameters to make the spectograms could possibly yield improved results. I would have liked to try fastai2 and some other variations on resnet architectures (xresnet, seresnet, etc) as well as the efficientnets. If I have time, I will add them to the github repository.

I strongly recommend reviewing the solutions in this Kaggle competition:
https://www.kaggle.com/c/freesound-audio-tagging-2019/discussion

I wasn't able to get to processing the audio files on GPU because I started the competition late.
https://towardsdatascience.com/how-to-easily-process-audio-on-your-gpu-with-tensorflow-2d9d91360f06

Another useful source suggests that the EfficientNet family of models may be useful:
https://towardsdatascience.com/sound-based-bird-classification-965d0ecacb2b
