pip install --upgrade pip "setuptools<70.0.0" wheel 
# TODO, unpin setuptools when this issue in flash attention is resolved
pip install torch==2.5.1 torchvision==0.20.1 --index-url https://download.pytorch.org/whl/cu121
pip install packaging
pip install flash-attn==2.7.2.post1 --no-build-isolation
pip install -r requirements.txt
pip install -e .
python -m nltk.downloader punkt