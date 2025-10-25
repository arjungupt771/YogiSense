# cell-A
!pip install kaggle

# Upload kaggle.json file from your Kaggle account (Account > API > Create New Token)
from google.colab import files
files.upload()  # upload kaggle.json here

!mkdir -p ~/.kaggle
!cp kaggle.json ~/.kaggle/
!chmod 600 ~/.kaggle/kaggle.json

# Download Yoga Dataset
!kaggle datasets download -d niharika41298/yoga-poses-dataset
!unzip yoga-poses-dataset.zip -d yogi_dataset
