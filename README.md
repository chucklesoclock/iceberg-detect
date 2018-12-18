# iceberg-detect
Capstone project for the Udacity Machine Learning Engineer Nanodegree.

This project aims to automatically discriminate between icebergs and ocean-going vessels in satellite radar data. 

## Relevant Files
- [Main Project Jupyter Notebook](project_notebook.ipynb)
- [Project Report PDF](/capstone_report/capstone_report.pdf)

## Gathering Dataset
Download from the [Kaggle competition page](https://www.kaggle.com/c/statoil-iceberg-classifier-challenge/data). WARNING: 1.5 GB size.

## Language
**Python 3.6.4**

### Software Libraries
- Keras
- TensorFlow
- TensorFlow-GPU
- SciKit-Learn
- Pandas
- NumPy
- SciPy
- Jupyter Notebook
- cudnn
- tqdm-keras

### Visualization Libraries
- Seaborn
- Plotly

### Full list
```
$ conda list
# packages in environment at C:\Users\[redacted]\Miniconda3\envs\cnn:
#
# Name                    Version                   Build  Channel
asn1crypto                0.24.0                   py36_0
astroid                   1.6.3                    py36_0
backports                 1.0              py36h81696a8_1
backports.weakref         1.0rc1                   py36_0
bleach                    1.5.0                    py36_0
ca-certificates           2018.03.07                    0
certifi                   2018.1.18                py36_0
cffi                      1.11.4           py36hfa6e2cd_0
chardet                   3.0.4            py36h420ce6e_1
colorama                  0.3.9            py36h029ae33_0
cryptography              2.1.4            py36he1d7878_0
cudatoolkit               8.0                           3
cudnn                     6.0                           0
cycler                    0.10.0           py36h009560c_0
decorator                 4.2.1                    py36_0
entrypoints               0.2.3            py36hfd66bb0_2
freetype                  2.8                  h51f8f2c_1
graphviz                  2.38.0                        4
h5py                      2.7.1            py36he54a1c3_0
hdf5                      1.10.1               h98b8871_1
html5lib                  0.9999999                py36_0
icc_rt                    2017.0.4             h97af966_0
icu                       58.2                 ha66f8fd_1
idna                      2.6              py36h148d497_1
intel-openmp              2018.0.0             hd92c6cd_8
ipykernel                 4.8.2                    py36_0
ipython                   6.2.1            py36h9cf0123_1
ipython_genutils          0.2.0            py36h3c5d0ee_0
ipywidgets                7.1.2                    py36_0
isort                     4.3.4                    py36_0
jedi                      0.11.1                   py36_0
jinja2                    2.10             py36h292fed1_0
jpeg                      9b                   hb83a4c4_2
jsonschema                2.6.0            py36h7636477_0
jupyter                   1.0.0                    py36_4
jupyter_client            5.2.2                    py36_0
jupyter_console           5.2.0            py36h6d89b47_1
jupyter_core              4.4.0            py36h56e9d50_0
jupyterlab                0.31.8                   py36_0
jupyterlab_launcher       0.10.2                   py36_0
keras                     2.1.5                    py36_0
keras-tqdm                2.0.1                     <pip>
lazy-object-proxy         1.3.1            py36hd1c21d2_0
libpng                    1.6.34               h79bbb47_0
libprotobuf               3.4.1                h3dba5dd_0
markdown                  2.6.9                    py36_0
markupsafe                1.0              py36h0e26971_1
matplotlib                2.1.2            py36h016c42a_0
mccabe                    0.6.1            py36hb41005a_1
mistune                   0.8.3                    py36_0
mkl                       2018.0.1             h2108138_4
nbconvert                 5.3.1            py36h8dc0fde_0
nbformat                  4.4.0            py36h3a5bc1b_0
nodejs                    8.9.3                hd6b2f15_0
notebook                  5.4.0                    py36_0
numpy                     1.12.1           py36hf30b8aa_1
openssl                   1.0.2o               h8ea7d77_0
pandas                    0.22.0           py36h6538335_0
pandoc                    1.19.2.1             hb2460c7_1
pandocfilters             1.4.2            py36h3ef6317_1
parso                     0.1.1            py36hae3edee_0
patsy                     0.5.0                    py36_0
pickleshare               0.7.4            py36h9de030f_0
pip                       9.0.1            py36h226ae91_4
pip                       9.0.3                     <pip>
plotly                    2.4.0                    py36_0
prompt_toolkit            1.0.15           py36h60b8f86_0
protobuf                  3.4.1            py36h07fa351_0
pycparser                 2.18             py36hd053e01_1
pydot                     1.2.4                    py36_0
pygments                  2.2.0            py36hb010967_0
pylint                    1.8.4                    py36_0
pyopenssl                 17.5.0           py36h5b7d817_0
pyparsing                 2.2.0            py36h785a196_1
pyqt                      5.6.0            py36hb5ed885_5
pysocks                   1.6.7            py36h698d350_1
python                    3.6.4                h6538335_1
python-dateutil           2.6.1            py36h509ddcb_1
pytz                      2018.3                   py36_0
pywinpty                  0.5              py36h6538335_2
pyyaml                    3.12             py36h1d1928f_1
pyzmq                     16.0.3           py36he714bf5_0
qt                        5.6.2           vc14h6f8c307_12
qtconsole                 4.3.1            py36h99a29a9_0
requests                  2.18.4           py36h4371aae_1
scikit-learn              0.19.1           py36h53aea1b_0
scipy                     1.0.0            py36h1260518_0
seaborn                   0.8.1            py36h9b69545_0
send2trash                1.5.0                    py36_0
setuptools                38.5.1                   py36_0
simplegeneric             0.8.1                    py36_2
sip                       4.18.1           py36h9c25514_2
six                       1.11.0           py36h4db2310_1
sqlite                    3.22.0               h9d3ae62_0
statsmodels               0.8.0            py36h6189b4c_0
tensorflow                1.2.1                    py36_0
tensorflow-gpu            1.1.0               np112py36_0
terminado                 0.8.1                    py36_1
testpath                  0.3.1            py36h2698cfe_0
tornado                   4.5.3                    py36_0
tqdm                      4.19.4           py36h02a35f0_0
traitlets                 4.3.2            py36h096827d_0
urllib3                   1.22             py36h276f60a_0
vc                        14                   h0510ff6_3
vs2015_runtime            14.0.25420                    0
wcwidth                   0.1.7            py36h3d5aa90_0
webencodings              0.5.1            py36h67c50ae_1
werkzeug                  0.14.1                   py36_0
wheel                     0.30.0           py36h6c3ec14_1
widgetsnbextension        3.1.4                    py36_0
win_inet_pton             1.0.1            py36he67d7fd_1
wincertstore              0.2              py36h7fe50ca_0
winpty                    0.4.3                         4
wrapt                     1.10.11          py36he5f5981_0
yaml                      0.1.7                hc54c509_2
zlib                      1.2.11               h8395fce_2
```
