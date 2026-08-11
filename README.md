markdown
# NightNext

Official code for NightNext nighttime semantic segmentation (Submitted to Pattern Analysis and Applications)

## Installation

Step 1. Clone and enter the repo:
```bash
git clone https://github.com/hxm1129/nightnext.git
cd nightnext
Step 2. Install dependencies:
bash
pip install -r ./requirements.txt
This installs the runtime, optional, and testing dependencies referenced from requirements/.
Step 3. Install MMSegmentation (this repo already contains the mmseg package, version 0.24.1):
bash
pip install -e .
Step 4. (Recommended) Verify the environment:
bash
python -c "import mmseg; print(mmseg.__version__)"
# expected: 0.24.1
Datasets
Experiments are conducted on two public nighttime segmentation datasets:
NightCity: https://dmcv.sjtu.edu.cn/people/phd/tanxin/NightCity/index.html
ACDC: https://acdc.vision.ee.ethz.ch/
Download the datasets and organize them as follows (modify data_root in the config files if your path differs):
plaintext
data/
├── nightcity/
│   ├── leftImg8bit/   # images
│   └── gtFine/        # annotations
└── acdc/
    ├── rgb_anon/      # images
    └── gt/            # annotations
Training
Train on NightCity:
bash
python tools/train.py myconfigs/nightnextconfig/7nightnext2.py --work-dir ./work_dirs/nightcity_exp
Train on ACDC Night:
bash
python tools/train.py myconfigs/nightnextconfig/2acdc.py --work-dir ./work_dirs/acdc_exp
Optimizer: AdamW, lr = 6e‑5, weight decay = 0.01, poly schedule.
Batch size: 4 per GPU. Total iterations: 80,000.
Pre‑trained weights: initialize the backbone with the SegNeXt MSCAN‑S Cityscapes‑pretrained weights (see myconfigs/_base_/models/mscan.py).
Testing
Test with the trained checkpoint:
bash
python tools/test.py myconfigs/nightnextconfig/7nightnext2.py <path‑to‑checkpoint.pth>
Quantitative metrics will be printed to the terminal.
Results are reported as mean ± std over three independent training runs.
Pretrained Weights
Checkpoints and pretrained backbone weights are stored under nightcitypth/ and pretrained/ (please place your trained .pth files there or update the config accordingly).
Troubleshooting
FileNotFoundError: requirements.txt: Ensure you run commands under the repository root directory.
Dataset loading error: Check the data_root variable in config files matches your local dataset path.
Acknowledgements
This project is based on MMSegmentation and SegNeXt.
Contact
For any other inquiries, please contact the corresponding author.
