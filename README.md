# deepfake-detection-for-image-video
deepfake detection for image/video



  ## Instructions for Usage
1) The main python file to run is detect_from_video.py in classification folder
2) The main file imports/uses two other python files, models.py and xception.py in network folder
3) Make sure you install all the dependencies in a new environment(for clarity)
4) In terminal, while running the detect_from_video.py, the line takes in the videopath arguments. (python <file> --videopath <path>)
5) In the main method in the file, after parsing the video path, compiler asks to select one of the architecture to test on,select one of them. (1.Xception, 2.EfficientNet)
6) Wait for the results for the selected option.

```
  python3 detect_from_video.py --video_path <path to test set>
```
## Results


| ![real](REAL.png) | 
|:--:| 

| ![fake](FAKE.png)| 
|:--:| 


