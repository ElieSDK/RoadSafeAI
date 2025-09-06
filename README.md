# RoadSafeAI: AI-Powered Road Damage Analysis

RoadSafeAI is an AI-powered system for analyzing road surface conditions and generating instant reports. It uses a multi-task model to classify street surface types and surface quality from images. The system supports imbalanced datasets, early stopping, and outputs confusion matrices and accuracy plots. The trained model can also be used to generate predictions on new road images.

---

## Requirements
- Python 3.8+
- PyTorch
- torchvision
- torchsampler
- pandas
- pillow
- matplotlib
- seaborn

There is also a Streamlit web application in the `streamlit` folder that requires its own `requirements.txt` for running the app.

---

## Dataset
The dataset used for training can be downloaded from [Zenodo](https://zenodo.org/records/11449977).

---

## Pretrained Model
The output of the trained model can be downloaded from [Hugging Face](https://huggingface.co/esdk/my-efficientnet-model/tree/main).

---

## How to Run Training

1. Clone the repository:
```bash
git clone https://github.com/eliesdk/roadsafeai.git
cd my_project
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Mount your Google Drive (if using Colab):
```python
from google.colab import drive
drive.mount('/content/drive')
```

4. Update the paths in `src/train.py`:
```python
CSV_PATH = "/content/drive/MyDrive/data/streetSurfaceVis_v1_0.csv"
IMG_DIR  = "/content/drive/MyDrive/data/s_1024"
```

5. Run training:
```bash
python src/train.py
```

6. Results:
- Best model checkpoint: `best_model.pt`
- Confusion matrices for validation set
- Accuracy per epoch plot

---

## Streamlit Web App
The project includes a Streamlit web application available at [https://roadsafe.streamlit.app/](https://roadsafe.streamlit.app/). The `streamlit` folder contains:
- `app.py` → Streamlit app code
- `city.csv` → Sample input data for the app
- `requirements.txt` → Dependencies for running the web app

Run the app locally with:
```bash
streamlit run streamlit/app.py
```

---

## File Structure

- `src/` → Contains the four main Python files: `dataset.py`, `model.py`, `utils.py`, `train.py`
- `streamlit/` → Contains Streamlit web app files
- `README.md` → Project documentation
- `requirements.txt` → Project dependencies
