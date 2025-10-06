_readme:
	echo << EOM
	Please see the README for project information.

	----------------------------------------------------------------
	Functions available in this project:

	clean				Removes the current model links and any intermediate files
						used in training models.

	clean-identify		Removes the current model softlink and any intermeidate
						files used in training the identification model.

	clean-classify		Removes the current model softlink and any intermeidate
    					files used in training the classification model.

	generate-data		Generates the training and testing data needed to train and
						evaluate the models.

	train-identify		Runs the training algorithm for the identification model.
						More information to follow once the idenfitication model
						is designed.

	train-classify		Runs the training algorithm for the classification model.
    					Once the model stops making significant improvements,
						updates the current model softlink and generates an accuracy
						report.

	hls4ml-translate	Convert the generated models for use and testing in FPGAs or
						emulation software for evaluation of success.

	----------------------------------------------------------------
	All Functions:

	EOM

clean:

clean-identify:

clean-classify:

generate-data:

train-identify:

train-classify:

hls4ml-translate:


