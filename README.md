# Surveillance-system
This program consists of a surveillance system capable of recognizing profiles and raising an alarm to notify that a given profile has been identified.It uses a trained dataset model with fair amount of accuracy.It shows the names of the people in dataset when one is recognized by camera while for others its shows "Unknown".
Firstly, I've made a dataset using Haarcascade Frontal face classifier which makes profiles of people who need to be recognized.Next,Training the model using the dataset with the help of opencv LBHP facerecognizer and numpy.Next is recognizing the known profiles among unknown profiles.After identification,i've used a arduino to raise an alarm using buzzer and the program also captures screenshots of the real time when a profile is recognized.
