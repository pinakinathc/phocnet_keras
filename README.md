Hi Thank you for viewing this repository.

This code implements the paper by Sebastian Sudholt, Gernot A. Fink,
Christened "PHOCNet: A Deep Convolutional Neural Network
for Word Spotting in Handwritten Documents"

The paper can be found at: https://arxiv.org/pdf/1604.00187.pdf

I shall write a few steps down for you to run this project in your machine.
Please note that I shall consider that you have git, pip3, and virtualenv.

Steps:
* Setup a new virtualenv using: `virtualenv -p python3 phocnet_keras`
* Install some essential packages using:
	- `pip3 install numpy`
	- `pip3 install pandas`
	- `pip3 install opencv-python`
	- `pip3 install tensorflow-gpu` (or if you do not have a GPU then, `pip3 install tensorflow`)
	- `pip3 install keras`
* Now, Clone this repository using `git clone https://github.com/pinakinathc/phocnet_keras`
* Go to the directory of project: `cd phocnet_keras`
* Now, untar the dataset present in `word` & `xml` folders using:
	- `tar -xvf words/words.tgz`
	- `tar -xvf xml/xml.tgz`
* We are now ready to execute the model. Execute: `python phoc.py`

Please note, if you do not have a GPU in your computer, you should comment the following lines:
- phoc_classifier.py => lines: {13-17}, 19, 75

If you have a GPU but do not have multiple GPUs in your system, please comment like:
- phoc_classifier.py => line: 75

I have not completed training, hence my model has an MAP of only 62% whereas the original paper claims to have map of 72.51%.