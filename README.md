# Sparse Pattern INDexing toolkit


## Installation
First, install some necessary tools. For CentOS users, just use the following command,

`sudo yum install -y git wget bzip2`

Then download SPIND package, 

`git clone https://github.com/LiuLab-CSRC/indexing.git`

### Install CrystFEL-SPIND
To integrate SPIND into CrystFEL, we add SPIND module to CrystFEL-0.6.2. See detailed installation instructions [here](http://www.desy.de/~twhite/crystfel/install.html).

### Install SPIND
SPIND is implemented using Python, you do not have to do standard `configure`, `make`, `make install`, just install necessary dependencies.

SPIND package has the following dependencies,

* numpy
* scipy
* h5py
* mpi4py
* yaml
* docopt

To make life easier, [anaconda](https://anaconda.org) is highly recommended for Python-related library management. 

Download `anaconda` for Python2.7 by

`wget https://repo.continuum.io/archive/Anaconda2-5.0.1-Linux-x86_64.sh`

Install anaconda, 

`bash Anaconda2-5.0.1-Linux-x86_64.sh`

Accept the license then wait minutes for installation.

After the installation, make the installer prepend anaconda location to `PATH` in your `.bashrc` file.

Then `source ~/.bashrc`.

Finally, install SPIND dependencies using

`conda install mpi4py yaml docopt`

Installation is done! Congratulations!