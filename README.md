## Guess the Engine (Piston Classifier)

A Machine Learning-based Streamlit app that classifies piston images as either perfect or defected.
The model is trained on a dataset of piston images with two quality categories and integrated into a simple, interactive web interface.

## About Machine Learning

Machine Learning is a branch of Artificial Intelligence that enables computers to learn from data and make predictions without being explicitly programmed.
In this project, the ML model learns to identify visual features of pistons such as texture, color, and surface damage patterns from hundreds of labeled images.
When a user uploads a new piston image, the model :
1. Converts the image into numerical pixel data.
2. Processes it using a Convolutional Neural Network (CNN).
3. Predicts whether the piston is Defected or Perfect.
This approach is widely used in industrial quality inspection, where automated visual checks are faster and more consistent than manual inspection.

## Libraries Used

1. TensorFlow : For building and training the deep learning (CNN) model.
2. Streamlit : For creating the interactive web interface.
3. NumPy : For numerical operations and image data manipulation.
4. Pillow (PIL) : For reading and processing uploaded images.
5. gdown : For automatically downloading the .h5 model file from Google Drive.

## How to Run
1. Clone the repository
```bash
   git clone https://github.com/kainewton/guess-the-engine.git
cd guess-the-engine
```
2. Install dependencies:
```bash
pip install -r requirements.txt
```
3. Run the streamlit
```bash
streamlit run app.py
```
## Model File
The trained model (.h5) is stored on google drive and will be automatically downloaded when the app runs :
```bash
https://drive.google.com/file/d/1oaJ0Uavz_0YDyhrwUSgcjN0gbsUaBGjI/view?usp=drive_link   
```




