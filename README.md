![ACM Research Banner Light](https://github.com/ACM-Research/paperImplementations/assets/108421238/467a89e3-72db-41d7-9a25-51d2c589bfd9)

# Spring 2026 Paper Implementation #2 - Mahd

# Setup

First, you'll need to clone the repoistory and specifically this branch. How I do it is in VS Code after opening an empty folder, in the command line I type "git init", "git remote add origin https://github.com/purvajpatel/acm-research-sp26", and then do "git pull origin main:Mahd2".

After doing so, you'll need to grab the dataset I'm using. I'm using the following: https://zenodo.org/records/7859211. From it, you'll need the metadata.zip download and also the training_mixed_set.zip file. unzip the training_mixed_set.zip into ModelCode, and retitle the new folder name "datasetOg". Make sure that within the dataset folder isn't an embedded folder but rather it contains all the images. Then unzip the metadata folder to ModelCode as well, make sure its name remains as "metadata". Again, also make sure inside isn't an embedded folder but rather all the csv files and stuff directly.

You'll also have to install a couple python libraries. To do this, you can just type in the terminal "pip install -r requirements.txt", and it should install. If it doesn't, then type in the terminal "pip install pandas tensorflow numpy transformers Pillow tensorflow.keras".

Now, run "AltModelRunner.ipynb" (NOT 'ModelRunner.ipynb') in the ModelCode folder, and that should create the model.

The next steps require having an ESP32-S3-WROOM-1 (the actual device) alongside a OV 2640 camera with you, so this is as far as you can go without that hardware. If you do have that ready, here's how you'd upload it to the esp:

First, the framework i'm using is ESP-IDF. You'll have to install the esp idf extension on VS Code, and make sure to follow the instructions carefully. Here's a guide on how to do that: https://github.com/espressif/vscode-esp-idf-extension/blob/master/README.md. You probably want to use the idf as its setup in the EspIdfCode folder though rather than making a new project, cause there's 2 external idf components (which are basically libraries) imported that otherwise you'd have to import manually

Next, find the "convertModelToArr.py" file in the "EspIdfCode" directory, pass in the path of the newly created model if it's not correct, then put the outputted .cpp file in EspIdfCode/main.

Next, navigate your vs code folder to the "EspIdfCode" directory. Make sure the esp idf extension is open - you should be able to tell if at the bottom you see a couple buttons like build, flash, and clean. Now, connect your ESP to the computer using a usb, make sure in "main.c" that the boolean for "isSendingImage" is true, click the build button to build the code and get it ready to be put on the esp, click the flash button to send the code to the esp, and now the esp should be running the code! To get results and see the camera work, go to "runModelRequest.ipynb", MAKE SURE THE COM PORT IS CORRECT, and click run to get the image used and the results of the model! It saves the first image in frame_0.png, and the 2nd in frame_1.png. The idea is you'd point it at a non object (like the ceiling) for one of the captures, and then an object, and see the difference in predictions.

# Paper: TinyOL: TinyML with Online-Learning on Microcontrollers

Link to the paper: https://arxiv.org/pdf/2103.08295  

Note that my code is more of a direct implementation of the ideas present in this paper towards my actual research project, rather than implementing the paper itself. I wasn't able to get the continual implementation done (for reasons that will be explained a bit later), but by the time my implementation's done, I will have implemented a cnn model onto a robot that was trained externally and have interfaces it with the camera.

The paper discusses the idea of implementing continuous learning on tiny microcontrollers, and particularly attempts to do this by treating the original mdoel as frozen, and adding on additional layers that actually weights that may be modified, allowing the model to be trained without causing an explosion in memory that comes when training on a small chip.

# Method

For this method, the first thing I needed to do was create a model to perform a task that is initially trained in Python. For the purposes of my project, the task at hand is basically to determine if there's a hazard immedietly up ahead, and avoid going there. For this, I used a CNN. The microcontroller i got, the ESP32 S3, came integrated with the OV 2640 camera attached onto it, so I had to make sure to set it up in code and that it could both read frames and send them to another device that could read them (in this case, that was sending them over UART serial communication towards the computer where a program would catch the serial communication).

Next, while normally to save quality I'd actually go for a lower camera quality, the OV2640's lowest camera quality was 96 x 96, so that would be the starting input for the model that i built. About building the model, I found a semi-decent dataset that was basically a video of a robot going through hallways, and occasionally running into hazards and other objects (such as people). The video was then split into images capturing each frame, and lableed depending on the object present (or lack of object present)

One huge challenge I ran into here was the model overfitting to specifically just the images presented, and not learning anythign meaningful. This likely happened because a lot of the images are just a few frames apart, so there's not much difference, and for a given segment of coming across a single object, there could be 30-40 images that are all quite similar to each other. So, even though it seemed like I got a lot of good data with around 8k images for training total initially, the issue was that a lot of the images were similar to each other, so it would just overfit and only perform on images very similar to those the dataset was based on. To account for this, I had to implement an alogrithm to loop through the images, and either only pick them whenever the label changed (so we were like 'switching clips' to another object) or when a lot of time has passed since the last picking.

Another thing I had to do was quantize the model to int8_t. The reason why this must be done is to save on memory; normally, you'd store model weights, biases, and such as floats. However, floats take 4 bytes while int8_t takes 1 byte, so if you can convert the weights and biases and other activation algorithms to just be stored as 8 bit integers, you save on 1/4 the memory, important when MCUs have very constrained memory.

Finally, after saving the model, I needed to figure out how to actually run it on the MCU. I found that you can store the entire model (weights as well as the actu layout of the model) as a C array, so i created a function to take in the .tflite model and convert it to such an array. Next, we needed to use a library for model inferencing. I initially was going to use tflite-micro for Arduino framework on the ESP32, but I ran into a great deal of import problems with it. The old code is in "EspOldCode". Therefore, I decided to switch to the esp-idf framework, though still using the tflite micro library (I initially tried to use the esp-dl library instead, but it wasn't very compatible with TensorFlow). 

One other thing to mention is that obviously, the main thing is that there's no continual learning done in this project like there is in the paper. This is something I'll definietely need to do in the paper, however I decided not to for this 2nd implementation. The main reason is because I initially had the assumption that ML libraries for MCUs had support for training models as well or at least adding non-frozen layers for training, but this is not true - at least as far as I can find, nearly all the ML libraries that are compatible with the ESP32 only support inferencing with no training support. This means that for training, I'd have to implement the forward and backward pass myself, which I felt would take too much time. Even if I were to look into the paper's Github and see how they implemented it, since they used a different MCU (an arduino nano), I'd have to adapt their code to the ESP32. I will likely come back in my own time to implement continual training, so I don't leave my mentees in the dark.

Finally, I set up class 1 to be object detected up ahead, and class 0 to be object not detected. I ran it for 30 epochs, and used a CNN with 2 Conv2D layers, which would then be flattened and go to a hidden layer before going to the output layer.

# Results And Evaluation

There are two aspects for the results: first, how the general model performed with its test set, and how the model ended up perofrming on the ESP32 when interfaced with the hardware camera. First I'll cover the model with the test set.

Initially as I mentioned, I attempted to include a lot of images in the training when a lot of them were similar to each other. This version ended up performing very well, with the following results below:

Confusion Matrix:
 [[1163  438]
 [ 195 1443]]
Accuracy:
 0.8045693115158999
Recall:
 0.8809523809523809
F1 Score:
 0.8201193520886616

 However, the main issue was that I hadn't realized it, but the model was heavily overfitting. Specifically, when I take any given frame from the camera, it always (I couldn't find a single counterexample) predicts class 1. Not just on the esp32, but when I ran the model in Python on my computer with an image captured from the esp, I got this same issue.

 So after I went back and fixed it, the model now had a lot less data (only about 2k points), but this time it didn't seem to overfit. It was at least learning something meaningful, as the accruacy was over 60% rather than being at excactly 50%. Here were the results:

Confusion Matrix:
 [[26 47]
 [14 49]]
Accuracy:
 0.5514705882352942
Recall:
 0.7777777777777778
F1 Score:
 0.6163522012578616

It can be seen that the model did have a high bias towards class 1, but with class 0 it at least predicted correctly some times.

When running it on the ESP32, I found that the model worked at least some of the time. Generally, when you point it at something without any distrinct object, the probability of class 0 gets to around 50%. When you point it at a distrinct object, the probability of class 1 is around 80-90%. The bias towards class 1 did continue and was very heavy, but the esp was able to utilize its camera and the model towards something meaningful.

We can see with the "runModelRequest.ipynb", the current output on that notebook showcases what I just mentioned. We first start off with an image of my bed with nothing on it, though a bit pointed at the ground so it doesn't detect the bedframe as an object. It still technically leans toward class 1, but the probability of class 0 is almsot half. Then, I put on the bed a bowl and face the camera towards it. We find next that it highly predicts class 1 as we expect.

# Novelty and Conclusion

Overall, the project itself isn't quite novel yet in this implementation - there have very likely existed proejcts that take in camera input and use a model on the chip to detect hazards up ahead. However, when we implement continual learning, the project should become a lot more novel, as generally continual training is underresearched in regards to embedded devices. Ideally, we would want to do something more complex with continual training rather than just a few non-frozen layers being slapped on, but even if we are unable to, it is still quite novel to have continual learning done on a robot to detect hazards (at least, I couldn't find any projects or products that already do this).

In conclusion, this implementation taught me a lot about using ML on embedded devices, gave me greater skill in using hardware in general, and taught me the importance of preprocessing the dataset to actually help the model learn meaningful information. Next steps would be to try implementing basic continual training, and thinking about how many more datasets would be needed to make the model robust. Finally, there should be better implementation for metrics on the esp device, including latency, memory usage, accuracy, and energy efficiency if possible.