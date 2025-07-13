# Cat-vs-Dog-Voice-recognition
* Meow vs Wofff!

# Status so far

* Load the wav files in from a root directory.
* Function to check for the total length of the two classs in seconds and their bitrates.
* Tried to get rid of long pauses using Energy VAD (Doesnt work and not sure if we should follow this approach). 
* If we find the average length of all the wavs, maybe we can randomly sample that duration from different instances of all the examples and then do a Mel-Coeficient analysis to generate the final feature vectors?
 -Update: Found the Mel-coeficients using a fixed frame length ad overlap sample. Maybe we can look into the average length after an initial training and validation of our model.

# To-Do(feel free to add all your thoughts and suggestions)

*There seems to also be a mfcc function built into librosa I an not sure however what other preprocessing is in there (eg preemphasizing):  
    mfccs = librosa.feature.mfcc(y=audio, sr=22050, n_mfcc=40)
    mfccs_scaled = np.mean(mfccs.T, axis=0)
* The total duration of cats are double than dogs. Find possible mitigations for that.
* Implement a noise redutction function and save the cleaned audio files in a folder in the repo.(Can also be used directly in the pipeline without saving to reduce complexity and save space!) Instead of noise reduction, we could also add gaussian white noise to all samples, that would also negate the background noise influence something like this maybe:
* Generating white noise file (has to be done for every new file, else all files would have the same 
pip install numpy
pip install scipy

import numpy as np
from scipy.io.wavfile import write

# Parameters
sample_rate = 44100  # Sample rate in Hz
duration = 5  # Duration of the white noise in seconds

# Generate white noise
noise = np.random.normal(0, 1, sample_rate * duration)

# Normalize the white noise
noise = noise / np.max(np.abs(noise))

# Convert the white noise to a 16-bit format
noise = (noise * 2**15).astype(np.int32)

# Save the white noise as a .wav file
write('white_noise.wav', sample_rate, noise

*add noise to cat/dog wav
import numpy as np
from scikits.audiolab import wavread, wavwrite

data1, fs1, enc1 = wavread("file1.wav")
data2, fs2, enc2 = wavread("file2.wav") #this would be the noise file 

assert fs1 == fs2 #im not sure if we need those two, hopefully we manage to write the same sample rate into the noise file and the encoding hopefully also matches
assert enc1 == enc2
result = 0.9 * data1 + 0.1 * data2  #this would be noticable level of noise i believe, maybe we have to test (0.99 to 0.01)

wavwrite(result, 'result.wav')

* maybe we should add an amplitude averaging preprocessing, some of the wavs are pretty quiet // Normalization
* Research on the best NN architecture for training. This paper   Enhancing Audio Classification Through MFCC Feature Extraction and Data Augmentation with CNN and RNN Models by Karim Mohammed Rezau, Md. Jewe, Md Shabiul Islam, Kazy Noor e Alam Siddiquee, Nick Barua, Muhammad Azizur Rahman, Mohammad Shan-A-Khuda, Rejwan Bin Sulaiman, Md Sadeque Imam Shaikh, Md Abrar Hamim1 F.M Tanmoy, Afraz Ul Haque, Musarrat Saberin Nipun, Navid Dorudian, Amer Kareem  is quite nice. They also only used mfccs and tried out different CNNs and RNNs depending on dataset size. This could be really helpful for us.

# Additional python packages installed so far

* Desly- sounddevice, EnergyVAD , soundfile, librosa
* Marie- 


# Best practices

* Try to do a merge request as soon as you are done for the day or instance so that no work is lost.
* Check for changes in the main branch before starting work so that you do not waste time doing something the other person has already implemented. 
