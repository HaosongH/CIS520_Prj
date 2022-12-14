# Remember to install kaggle-api from here https://github.com/Kaggle/kaggle-api

# Download anime faces dataset from kaggle
kaggle datasets download splcher/animefacedataset -p ../datasets/anime_faces
unzip ../datasets/anime_faces/animefacedataset.zip -d ../datasets/anime_faces/
rm ../datasets/anime_faces/animefacedataset.zip

python torch_data.py