# Dependencies

```
tensorflow
tqdm
fmutils
gray2color
cv2
skimage
scipy
numpy
```
# Usage
Download the model from [here](https://drive.google.com/file/d/1-QMFaS8RS9VF242K2KnDR1QS1fGJI4h4/view?usp=sharing)
```
python slide_inference.py -m "path\to\model.h5" -sd "path\to\slides\" -dd "path\to\save\preds\" -b True -r False

```

## Help Message

```
    -h, --help            
        show this help message and exit. 
        Only model path and video path are required. 
    -m  --model_path
        path to the directory where all the images of sequence are.
    -sd --slide_dir
        directory where slides are.
    -dd --dest_dir
        directory where to write predictions.
    -b --blend
         Whether to overlay predictions over image or not.
    -r --draw_bdr
        Whether to draw borders or fill the nuclei detections.
   
```
