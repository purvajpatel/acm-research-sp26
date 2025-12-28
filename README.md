![ACM Research Banner Light](https://github.com/ACM-Research/paperImplementations/assets/108421238/467a89e3-72db-41d7-9a25-51d2c589bfd9)

# Spring 2026 Paper Implementations

# Setup

First, you'll need to clone the repoistory and specifically this branch. How I do it is in VS Code after opening an empty folder, in the command line I type "git init", "git remote add origin https://github.com/purvajpatel/acm-research-sp26", and then do "git pull origin main:Mahd".

After pulling, first you will need several .bin files. I didn't push them to github because they're large files and I thought using Git LFS would cause too big of a slow down, but you'll need to download 7 .bin files in total: 1 for the student model that facilitates the merging, and 6 for the specialized teacher models that will be merged. The steps are given below:

1. First, go to this URL https://huggingface.co/google-bert/bert-base-uncased/tree/main, and download the "pytorch_model.bin" file there. Put that into the StudentModel folder.
2. Go to this URL https://huggingface.co/textattack/bert-base-uncased-CoLA/tree/main, download the same "pytorch_model.bin" file, and put that into the TeacherModels/CoLA folder.
3. Go to this URL https://huggingface.co/textattack/bert-base-uncased-MNLI/tree/main, download the same file and put it into TeacherModels/MNLI.
4. Go to this URL https://huggingface.co/textattack/bert-base-uncased-MRPC/tree/main, download the same file and put it into TeacherModels/MRPC.
5. Go to this URL https://huggingface.co/textattack/bert-base-uncased-QNLI/tree/main, download the same file and put it into TeacherModels/QNLI.
6. Go to this URL https://huggingface.co/textattack/bert-base-uncased-QQP/tree/main, download the same file and put it into TeacherModels/QQP.
7. Go to this URL https://huggingface.co/textattack/bert-base-uncased-SST-2/tree/main, download the same file and put it into TeacherModels/SST-2.

Finally, you'll have to install a couple python libraries. To do this, you can just type in the terminal "pip install -r requirements.txt", and it should install. If it doesn't, then type in the terminal "pip install pandas torch numpy transformers".

Now, just run "placeholderMain.py" in the MainCode folder, and you should start to see results!

# Paper: Model Merging via Multi-Teacher Knowledge Distillation

Link to the paper: https://arxiv.org/pdf/2512.21288

Model merging is a novel concept in the realm of Large Language Models and NLP, which involves merging several models fine tuned from the same base model for a specific task into a merged model that is able to successfully do all those tasks. The purpose of model merging is to reduce memory costs by reducing redundant weights between the tasks, and also reducing inference costs with a single unified model for all tasks.

# Method

For this implementation, I decided to take several NLP tasks from the GLUE dataset, and use a separate base model from the paper; I used Bert-Base-Uncased with 6 fine tuned variants, as opposed to the paper which used GPT-2. I set up the implementation to merge the model weights using KL Divergence and Sharpeness Aware Minimization, with about 32 points for training per task, 8 points for validation per task, and 20 points for testing per task. After creating the merged model, I evaluated it on each task in terms of its accuracy and runtime.

We actually get to use unsupervised data for this, because our fine tuned models prior to merging can basically provide us with the labels that will allow us to calculate loss. Additionally, we do layer-wise merging, so we store the differences of the trained model weights against the fine tuned models at each layer rather than over all the layers. 

The purpose of using KL Divergence in our loss is to evaluate how different the weights of each fine tuned model are from the merged model, basically seeing how well they will still do on their task. KL Divergence measures the difference in two probability distributions, which in this case are the merged model and the fine tuned models, which is a better measure of loss during merging compared to other loss functions because of how it directly measures how different the models end up being.

Additionally, the algorithm uses Sharpness-Aware-Minimization in order to find minima that are more flattened rather than a minima at a very sharp curve. This is because minima that are more flattened are more generalized and less prone to drastic changes upon slightly tuning the model weights, as opposed to sharp minima which may be overfitting.

# Results And Evaluation

We can see that when running 6 tasks, I get the following results: 

Task 'cola': Agreement Rate=0.5000, Runtime=4.25s
Task 'mnli': Agreement Rate=0.6000, Runtime=4.55s
Task 'mrpc': Agreement Rate=0.7000, Runtime=4.17s
Task 'qnli': Agreement Rate=0.4000, Runtime=4.53s
Task 'qqp': Agreement Rate=0.4000, Runtime=4.26s
Task 'sst2': Agreement Rate=0.4500, Runtime=4.45s

Average Agreement Rate across all tasks: 0.5083

It doesn't seem like the model learns much meaningful, although the model weights are changing. MRPC and MNLI did above average, but that could be because the testing size was so small, so through random guessing it ended up doing better. My implementation is likely not correct and there's probably an issue regarding actually updating the weights that I'd need to fix. It coudl also be an issue in terms of epochs, but the paper used fewer epochs and had decent results.

In general for what would the advantages and disadvantages, The advantage of SAMerging is being about to build a merged model without requiring much data, and having it reduce overall memory and inference time withotu sacrificing too much accuracy.

The main disadvantage of this approach is that its a bit niche. It only applies to models that are fine tuned variants of the same base model, and also only applies if the tasks have some overlap with each other. Additionally, there hasn't been much testing of the model, so although the results of the paper were very good, more testing is required to evaluate it across several domains.

# Novelty and Conclusion

The novelty in this paper is providing a new way to merge models that drastically reduces the amount of data needed for model merging while maintaing higher accuracy and speed than most other merging models.

Overall, this research is very promising in being the definitive way in which to merge fine tuned models, but more research and testing should be done to see its limitations and other potential drawbacks in comparison to other models.

In particular, I would like to test it on various other base models when testing NLP tasks, and expand it to other natural language tasks such as video to text or image to text, and even beyond NLP tasks such as text to image or text to video. Additionally, it could be a potential idea to evaluate first if merging will be effective within a certain error based on the compatibility of the tasks, because typically merging is only effective when the tasks being merged into one model are similar to one another; otherwise the model becomes too averaged out to be good at any tasks.

Finally, one possibility of future work would be to see if the model merging could be epanding to not just merging fine tuned variants of the same base model, but also merging several specialized models with different bases into one model. The main reason this isn't done is because of the different dimensions in things such as parameters and layers that different base models use, but there could be a way to implement it.