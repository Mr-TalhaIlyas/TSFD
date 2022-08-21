# Download from Google Drive

To download the pre-trained weights of TSFD-Net click the following link,
### *[weights](https://drive.google.com/drive/folders/1XrPpOD9G3nbqWhDfjOynrLGGbEI06DDD?usp=sharing)*

## Evaluation

As explained in paper we evaluted the model by randomly splitting the data `[train/val:test]=[8:2]` ratio. We ran all experiments multiple times then averaged them to get the final results. The splits were made randomly each time after the data was processed as explained [here](https://github.com/Mr-TalhaIlyas/Prerpcessing-PanNuke-Nuclei-Instance-Segmentation-Dataset). As for testing the uplaoded weights (*for a specific split*) kindly split the data according to the file list given in `splits` dir in this repo after preprocessing and then run [`eval`](https://github.com/Mr-TalhaIlyas/TSFD/tree/master/eval) script. 😃






