# Setup on Ubuntu 18.04 (after clean install)

The following commands seem to build our simulation environment (tested on 2020/2/19).

```
$ sudo apt-get update
$ sudo apt-get install cmake libopenmpi-dev python3-dev zlib1g-dev python3-pip python3-tk git -y
$ pip3 install tensorflow==1.12.0 scikit-learn autograd GPy scipy matplotlib gym==0.12.1
$ cd ~
$ git clone https://github.com/openai/baselines.git
$ cd baselines
$ git checkout fa37beb52e867dd7bd9ae6cdeb3a5e64f22f8546
$ pip3 install -e .
$ pip3 uninstall gym -y
$ pip3 install gym==0.12.1
```

Actually, our code is used in simulation environments built more than six months ago.
Please see also the note below.

Recent version of numpy shows FutureWarning for this code.
In case of trouble, please downgrade numpy as follows.
```
$ pip3 uninstall numpy -y
$ pip3 install numpy==1.16.3
```

# Run pendulum

```
$ bash run_pendulum.bash
```
* when changing policy, please modify `policy_mode` in `run_pendulum.bash`
* when changing modeling error of PM, please modify `alpha2` in `pendulum_real_model.py`


# Run cartpole

```
$ bash run_cartpole.bash
```
* when changing policy, please modify `policy_mode` in `run_cartpole.bash`
* when changing modeling error of PM, please modify `cth` in `cartpole_real_model.py`


# Note
Our results and debugs are actually based on the following:
```
$ pip3 freeze
absl-py==0.7.1
apturl==0.5.2
asn1crypto==0.24.0
astor==0.7.1
autograd==1.2
-e git+https://github.com/openai/baselines.git@3301089b48c42b87b396e246ea3f56fa4bfc9678#egg=baselines
Brlapi==0.6.6
certifi==2019.3.9
chardet==3.0.4
Click==7.0
cloudpickle==0.8.1
command-not-found==0.3
cryptography==2.1.4
cupshelpers==1.0
cycler==0.10.0
decorator==4.4.0
defer==1.0.6
dill==0.2.9
distro-info===0.18ubuntu0.18.04.1
future==0.17.1
gast==0.2.2
GPy==1.9.6
grpcio==1.20.1
gym==0.12.1
h5py==2.9.0
httplib2==0.9.2
idna==2.8
joblib==0.13.2
Keras-Applications==1.0.7
Keras-Preprocessing==1.0.9
keyring==10.6.0
keyrings.alt==3.0
kiwisolver==1.1.0
language-selector==0.1
launchpadlib==1.10.6
lazr.restfulclient==0.13.5
lazr.uri==1.0.3
louis==3.5.0
macaroonbakery==1.1.3
Mako==1.0.7
Markdown==3.1
MarkupSafe==1.0
matplotlib==3.0.3
netifaces==0.10.4
numpy==1.16.3
oauth==1.0.1
olefile==0.45.1
opencv-python==4.1.0.25
paramz==0.9.4
pexpect==4.2.1
Pillow==5.1.0
progressbar2==3.39.3
protobuf==3.7.1
pycairo==1.16.2
pycrypto==2.6.1
pycups==1.9.73
pyglet==1.3.2
pygobject==3.26.1
pymacaroons==0.13.0
PyNaCl==1.1.2
pyparsing==2.4.0
pyRFC3339==1.0
python-apt==1.6.5+ubuntu0.2
python-dateutil==2.8.0
python-debian==0.1.32
python-utils==2.3.0
pytz==2018.3
pyxdg==0.25
PyYAML==3.12
reportlab==3.4.0
requests==2.21.0
requests-unixsocket==0.1.5
scikit-learn==0.20.3
scipy==1.2.1
SecretStorage==2.3.1
simplejson==3.13.2
six==1.12.0
system-service==0.3
systemd-python==234
tensorboard==1.12.2
tensorflow==1.12.0
termcolor==1.1.0
tqdm==4.31.1
ubuntu-drivers-common==0.0.0
ufw==0.36
unattended-upgrades==0.1
urllib3==1.24.3
usb-creator==0.3.3
wadllib==1.3.2
Werkzeug==0.15.2
xkit==0.0.0
zope.interface==4.3.2
```
