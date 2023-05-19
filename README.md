The repo consists of code to extract and transform tables in pdf to html format

## Installing libraries

### Layout Parser
pip install layoutparser
pip install "layoutparser[ocr]"
pip install "layoutparser[paddledetection]"
pip install layoutparser torchvision && pip install "git+https://github.com/facebookresearch/detectron2.git@v0.5#egg=detectron2"

### PaddleOCR 
git clone https://github.com/PaddlePaddle/PaddleOCR.git

### pypdfium2
pip install pypdfium2

### pdfminer
pip install pdfminer.six