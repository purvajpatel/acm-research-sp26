![ACM Research Banner Light](https://github.com/ACM-Research/paperImplementations/assets/108421238/467a89e3-72db-41d7-9a25-51d2c589bfd9)

# Spring 2026 Paper Implementation #2 - Mahd

# Setup

First, you'll need to clone the repoistory and specifically this branch. How I do it is in VS Code after opening an empty folder, in the command line I type "git init", "git remote add origin https://github.com/purvajpatel/acm-research-sp26", and then do "git pull origin main:Mahd2".

After doing so, you'll need to grab the dataset I'm using. I'm using the following: https://zenodo.org/records/7859211. From it, you'll need the metadata.zip download and also the training_mixed_set.zip file. unzip the training_mixed_set.zip into ModelCode, and retitle the new folder name "dataset". Make sure that within the dataset folder isn't an embedded folder but rather it contains all the images. Then unzip the metadata folder to ModelCode as well, make sure its name remains as "metadata". Again, also make sure inside isn't an embedded folder but rather all the csv files and stuff directly.

You'll also have to install a couple python libraries. To do this, you can just type in the terminal "pip install -r requirements.txt", and it should install. If it doesn't, then type in the terminal "pip install pandas torch numpy transformers Pillow".

Now, run "ModelRunner.ipynb" in the ModelCode folder, and that should create the model.

The next steps require having an ESP32-S3-WROOM-1 (the actual device) alongside a OV 2640 camera with you, so this is as far as you can go with that. If you do have that ready, here's how you'd upload it to the esp:

First, the framework i'm using is ESP-IDF. You'll have to install the esp idf extension on VS Code, and make sure to follow the instructions carefully. Here's a guide on how to do that: https://github.com/espressif/vscode-esp-idf-extension/blob/master/README.md.

Next, find the "convertModelToArr.py" file in the "EspIdfCode" directory, pass in the path of the newly created model if it's not correct, then put the outputted .h file in EspIdfCode/main.

Next, navigate your vs code folder to the "EspIdfCode" directory. Make sure the esp idf extension is open - you should be able to tell if at the bottom you see a couple buttons like build, flash, and clean. Now, connect your ESP to the computer using a usb, make sure in "main.c" that the boolean for "isSendingImage" is true, click the build button to build the code and get it ready to be put on the esp, click the flash button to send the code to the esp, and now the esp should be running the code! To get results and see the camera work, go to "runModelRequest.py", MAKE SURE THE COM PORT IS CORRECT, and click run to get the image used and the results of the model!

# Paper: TinyOL: TinyML with Online-Learning on Microcontrollers

Link to the paper: https://arxiv.org/pdf/2103.08295  

Note that my code is more of a direct implementation of the ideas present in this paper towards my actual research project, rather than implementing the paper itself. I wasn't able to get the continual implementation done (for reasons that will be explained a bit later), but by the time my implementation's done, I will have implemented a cnn model onto a robot that was trained externally and have interfaces it with the camera.

# Method

Blah Blah Blah

# Results And Evaluation

Blah Blah Blah

# Novelty and Conclusion

Blah Blah Blah