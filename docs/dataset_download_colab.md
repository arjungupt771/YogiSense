# Download the pose image dataset in Google Colab

These commands are for a Colab notebook cell, not a local Python script.

```python
!pip install kaggle
from google.colab import files
files.upload()  # Upload kaggle.json from your Kaggle account.
!mkdir -p ~/.kaggle
!cp kaggle.json ~/.kaggle/
!chmod 600 ~/.kaggle/kaggle.json
!kaggle datasets download -d niharika41298/yoga-poses-dataset
!unzip -q yoga-poses-dataset.zip -d data/source_archive
!mkdir -p data/raw_images
!mv data/source_archive/DATASET/* data/raw_images/
```
