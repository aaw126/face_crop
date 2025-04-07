# face_crop
A project for cropping faces in videos

  ⚙ Installation
Create conda environment:
  conda create -n crop python=3.10
  conda activate crop
  pip install --upgrade scenedetect[opencv]
Besides, ffmpeg is also needed:
  apt-get install ffmpeg

Before use, the model has been downloaded and is called face_landmarker_v2_with_blendshapes.task.The code processes video data in multiple processes. You can create ten folders to store the original video and ten output folders to store the cropped commands. You can also modify the number of processes in the code to set the processes randomly.
run with: bash parallel_run.sh.

You can also prepare the input and output folders for a single process by running:python extract_face_pipeline.py.

  
