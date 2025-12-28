![ACM Research Banner Light](https://github.com/ACM-Research/paperImplementations/assets/108421238/467a89e3-72db-41d7-9a25-51d2c589bfd9)

# Spring 2026 Paper Implementation #1

##   ☆ Project Overview
The model utilizes the deep neural network DeepSpeech2, which processes audio and predicts phonemes (the smallest possible phonetic unit that distinguishes words from one another). The study found that the model can be used to optimize cochlear implants to improve the clarity and understanding of spech signals being translated.

##  ☆ Motivation & Impact
The motivation behind the paper is to investigate how electrical encoding of speech-related information at the cochlea affects speech comprehension at the word and phoneme level.

## ☆ Novelty
The paper addresses how encoding of speech information affects comprehension of words and word parts as opposed to most other approaches where the focus is on the biology of the ear &  how the electrodes sent out by the CI are recieve to the body's neurons.

In addition, instead of using low-level acoustic features to improve speech/CI hearing quality, they use the model to transform unclear signals into decipherable audio  using knowledge of phonemes to distinguish words from one another.

## ☆ Advantages
 
 Methods that mainly take into account the connection between the ear and the cochlear implant are unable to model higher levels of auditory processing and understand the sound as human parts of speech, as well as how the audio is processed and percieved by the brain. 
 
 The model presented in the paper shows  the signals sent to the ear by the CI impacts how the speech is understood by the user by breaking down the sound into word parts that can be reconstructed as words and sentences. It is also made to process audio casually, the same way that humans do, meaning the result will be more akin to comprehensible speech than other models.

## ☆ Disadvantages
The model struggles with phoneme confusion, and the vocoder used to replication CI user input is not a completely accurate stand in.

## ☆ Resources

#### Paper Implemented: DeepSpeech models show Human-like Performance and Processing of Cochlear Implant Inputs

###### ⤷ Link: https://www.alphaxiv.org/abs/2407.20535?chatId=019b58b8-91bb-7f2f-98f8-d108ee60e060

#### Dataset Used: Librispeech (train-clean-100)
###### ⤷ Link: https://www.openslr.org/12