# ReNomIMG v1.0

ReNom IMG is model developing tool for object detection.

##### ReNomIMG - https://www.renom.jp/packages/renomimg/index.html

## Recommended Environment
- OS: Ubuntu 16.04
- Browser: Google Chrome(version 63.0.3239.132)
- Python: >=3.5

## Install
ReNomIMG requires ReNom.

If you haven't install ReNom, you must install ReNom from www.renom.jp.

## Install ReNomIMG package
Linux user can install ReNomIMG from Wheel package.

Other os user can't install from Wheel package but can install from source.

The Wheel package is provided at:

`https://grid-devs.gitlab.io/ReNomIMG/bin/renom_img-VERSION-cp35-cp35m-linux_x86_64.whl`

(VERSION is stands for actual version number e.g. 0.0.1)

You can install the wheel package with pip3 command::

`pip3 install https://grid-devs.gitlab.io/ReNomIMG/bin/renom_img-1.0.9-cp36-cp36m-linux_x86_64.whl`

## Install from source
For installing ReNomIMG, download the repository from following url.

`git clone https://github.com/ReNom-dev-team/ReNomIMG.git`

And move into ReNomIMG directory.
`cd ReNomIMG`

Then install all required packages.

`pip install -r requirements.txt`

And install renom module using following command.

`pip install -e .`

At last, build extension modules.

`python setup.py build`


## How to start

1.Type following command in ReNomIMG directory.

`python -m renom_img`

Or, following command is available from 0.7beta.

`renom_img`

Second command can be called in any folder and creates `datasrc` folder in current directory.
Please set your dataset to the created directory.


If the server starts, you will see a message like below.

<img src='./js/static/img/server_run.png' width='60%'/>

Then 'datasrc' folder will be created in your current directory.
Please set images and labels according to `2.Create dataset directory` description.


## How to use

The following videos describes how to use ReNomIMG.
In this video, the Oxford-IIIT Pet Dataset is used.

- Cats and Dogs Classification
https://github.com/JDonini/Cats_Dogs_Classification

- O. M. Parkhi, A. Vedaldi, A. Zisserman, C. V. Jawahar
Cats and Dogs
IEEE Conference on Computer Vision and Pattern Recognition, 2012
Bibtex
http://www.robots.ox.ac.uk/~vgg/data/pets/


#### Where to put your data. 
Please put image and label data to datasrc directory. The folder structure is below.

To create datasrc directory, please start ReNomIMG server according to 'How to start'.

The datasrc directory is automatically created when you start up server.

```
ReNomIMG
    └── storage
    |   └── test_database.db // Database(sqlite3).
    └── datasrc
        ├── img              // image for training and validation
        ├── label            // lable for training and validation
        └── prediction_set   // dataset for prediction
            ├── img          // image for prediction
            └── output       // results of prediction
                ├── csv      // annotation of prediction in csv formats
                └── xml      // annotation of prediction in xml formats

```

## Format of xml file.

The format of the xml file which created by ReNomTAG follows [PASCAL VOC] format.

An example is bellow.

```
<annotation>
 <folder>
  dataset
 </folder>
 <filename>
  2007_000027.jpg
 </filename>
 <object>
  <pose>
   Unspecified
  </pose>
  <name>
   cat
  </name>
  <truncated>
   0
  </truncated>
  <difficult>
   0
  </difficult>
  <bndbox>
   <ymax>
    203.02013422818794
   </ymax>
   <xmin>
    134.7902328154634
   </xmin>
   <xmax>
    238.81923552543284
   </xmax>
   <ymin>
    104.02684563758389
   </ymin>
  </bndbox>
 </object>
 <source>
  <database>
   Unknown
  </database>
 </source>
 <path>
  dataset/2007_000027.jpg
 </path>
 <segments>
  0
 </segments>
 <size>
  <width>
   486
  </width>
  <height>
   500
  </height>
  <depth>
   3
  </depth>
 </size>
</annotation>
```

## License

“ReNomIMG” is provided by GRID inc., as subscribed software.  By downloading ReNomIMG, you are agreeing to be bound by our ReNom Subscription agreement between you and GRID inc.
To use ReNomIMG for commercial purposes, you must first obtain a paid license. Please contact us or one of our resellers.  If you are an individual wishing to use ReNomIMG for academic, educational and/or product evaluation purposes, you may use ReNomIMG royalty-free.
The ReNom Subscription agreements are subject to change without notice. You agree to be bound by any such revisions. You are responsible for visiting www.renom.jp to determine the latest terms to which you are bound.

[PASCAL VOC]:http://host.robots.ox.ac.uk/pascal/VOC/
