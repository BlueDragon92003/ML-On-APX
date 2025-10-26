# Cluster Analysis ML Model for CMS L1

This is a undergraduate research project developed by Quentin Schultz
(qschultz@wisc.edu; quentin@theschultzfamily.us) under advisory by Sidhara Dasu
(dasu@hep.wisc.edu) and Abhi Mallampalli (amallampalli@wisc.edu).

The project goal is to develop an Machine-Learning (ML) model to identify and
classify calorimetry events within the L1 trigger system and see if it is more
efficient or more accurate than the current algorithmic method in use.

## Project Milestones

- [ ] Develop a model for classifying identified clusters between hadronic 
      (pions, etc.) and electromagnetic (electrons, photons, etc.) clusters. 
      Data is obtained from two sources, RCT with ECAL energy clusters at the
      crystal-level (0.087x0.087) resolution and HCAL at tower-level (0.35x0.35)
      resolution. These data need to be correlated and compared to make the
      classification. The lateral cluster size in ECAL and HCAL also has some
      discrimination power.
- [ ] Evaluate if the accuracy of the model is high enough to continue with this
        project, and perhaps retrain the model to reach desired accuracies.
- [ ] Develop a model for identifying those clusters and passing them to the next
        model in the expected format
- [ ] Evaluate if the accuracy of the model is high enough to continue with this
        project, and perhaps retrain the model to reach desired accuracies.
- [ ] Translate the models with HLS4ML and test their execution time in
        comparison to the current algorithm
- [ ] Evaluate if the models' efficieny is high enough to continue with this
        project, and perhaps retrain the model(s) to improve efficiency

## How to run this project

Most of this project can be manipulated using `make`.
Running `make` will give a complete overview of actions that can be
completed. Below is the summary of the top-level actions that can be completed:

**`clean`**: Removes the current model links and any intermediate files used in
    training models.

**`clean-identify`**: Removes the current model softlink and any intermeidate
    files used in training the identification model.

**`clean-classify`**: Removes the current model softlink and any intermeidate
    files used in training the classification model.

**`generate-data`**: Generates the training and testing data needed to train
    and evaluate the models, encompassing:

**`train-identify`**: Runs the training algorithm for the identification model.  
    More information to follow once the idenfitication model is designed.

**`train-classify`**: Runs the training algorithm for the classification model.  
    Once the model stops making significant improvements, updates the current
    model softlink and generates an accuracy report.

**`hls4ml-translate`**: Convert the generated models for use and testing in
    FPGAs or emulation software for evaluation of success.

