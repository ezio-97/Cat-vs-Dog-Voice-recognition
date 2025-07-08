# Cat-vs-Dog-Voice-recognition
* Meow vs Wofff!

# Status so far

* Load the wav files in from a root directory.
* Function to check for the total length of the two classs in seconds and their bitrates.
* Tried to get rid of long pauses using Energy VAD (Doesnt work and not sure if we should follow this approach).
* If we find the average length of all the wavs, maybe we can randomly sample that duration from different instances of all the examples and then do a Mel-Coeficient analysis to generate the final feature vectors?
 -Update: Found the Mel-coeficients using a fixed frame length ad overlap sample. Maybe we can look into the average length after an initial training and validation of our model.

# To-Do(feel free to add all your thoughts and suggestions)

* The total duration of cats are double than dogs. Find possible mitigations for that.
* Implement a noise redutction function and save the cleaned audio files in a folder in the repo.(Can also be used directly in the pipeline without saving to reduce complexity and save space!)
* Research on the best NN architecture for training.

# Additional python packages installed so far

* Desly- sounddevice, EnergyVAD , soundfile, librosa
* Marie- 


# Best practices

* Try to do a merge request as soon as you are done for the day or instance so that no work is lost.
* Check for changes in the main branch before starting work so that you do not waste time doing something the other person has already implemented. 