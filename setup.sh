python3 -m venv env
source env/bin/activate
pip install -r environment.txt
mkdir data
cd data
kaggle competitions download -c ieee-fraud-detection
cd ..
