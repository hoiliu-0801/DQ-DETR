pip install torch==2.4.0 torchvision==0.19.0 torchaudio==2.4.0 --index-url https://download.pytorch.org/whl/cu124

pip install cython
pip install -r requirements.txt
conda install -c conda-forge libstdcxx-ng

pip install cython==0.29.36
pip install "git+https://github.com/jwwangchn/cocoapi-aitod.git#subdirectory=aitodpycocotools"

cd models/dqdetr/ops
python setup.py build install
# unit test (should see all checking is True)
python test.py
cd ../../..

