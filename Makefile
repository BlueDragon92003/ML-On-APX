define readme
Please see the README for project information.

----------------------------------------------------------------
Functions available in this project:

install-dependencies	Installs relevant dependencies via pip

clean					Removes the current model links and any intermediate
						files used in training models.

run-tests				Runs the test suite to ensure non-model-training code
						behaves as expected.

clean-identify			Removes the current model softlink and any intermeidate
						files used in training the identification model.

clean-classify			Removes the current model softlink and any intermeidate
						files used in training the classification model.

generate-data			Generates the training and testing data needed to train and
						evaluate the models.

train-identify			Runs the training algorithm for the identification
						model. More information to follow once the
						identification model is designed.

train-classify			Runs the training algorithm for the classification
						model. Once the model stops making significant
						improvements, updates the current model softlink and
						generates an accuracy report.

hls4ml-translate		Convert the generated models for use and testing in
						FPGAs or emulation software for evaluation of success.


endef
export readme

_readme:
	@echo "$$readme"

install-dependencies:
	python -m ensurepip --upgrade
	python -m pip install torch==2.9.1 torchvision==0.24.1
	python -m pip install uproot=5.7.1
	python -m pip install numpy==2.3
	python -m pip install colour-runner==0.1.1
	python -m pip install argparse==1.4

clean: clean-identify clean-classify

run-tests:
	./bash/test-identification.sh
	./bash/test-classification.sh	

clean-identify:
	./bash/clean-identification.sh

clean-classify:
	./bash/clean-classification.sh

generate-data: data/PLACEHOLDER.csv
	@echo "This process is currently unsupported."

train-identify: generate-data
	python3 "./cluster-identification/main.py"

train-classify: generate-data
	python3 "./cluster-classification/main.py"

hls4ml-translate: models/current-classification.pth models/current-identification.pth
	@echo "This process is currently unsupported."

models/current-identification.pth: train-identify
	@echo "Training missing identification model before translation"

models/current-classification.pth: train-classification
	@echo "Training missing classification model before translation"
