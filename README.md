![ACM Research Banner Light](https://github.com/ACM-Research/paperImplementations/assets/108421238/467a89e3-72db-41d7-9a25-51d2c589bfd9)

# Spring 2026 Paper Implementations

# Paper: Model Merging via Multi-Teacher Knowledge Distillation

Model merging is a novel concept in the realm of Large Language Models and NLP, which involves merging several models fine tuned from the same base model for a specific task into a merged model that is able to successfully do all those tasks. The purpose of model merging is to reduce memory costs by reducing redundant weights between the tasks, and also reducing inference costs with a single unified model for all tasks.

# Method

For this implementation, I decided to take several NLP tasks from the GLUE dataset, and use a separate base model from the paper; I used Bert-Base-Uncased with 6 fine tuned variants, as opposed to the paper which used GPT-2. I set up the implementation to merge the model weights using KL Divergence and Sharpeness Aware Minimization, with about 32 points for training per task, 8 points for validation per task, and 20 points for testing per task. After creating the merged model, I evaluated it on each task in terms of its accuracy and runtime.

We actually get to use unsupervised data for this, because our fine tuned models prior to merging can basically provide us with the labels that will allow us to calculate loss. Additionally, we do layer-wise merging, so we store the differences of the trained model weights against the fine tuned models at each layer rather than over all the layers. 

The purpose of using KL Divergence in our loss is to evaluate how different the weights of each fine tuned model are from the merged model, basically seeing how well they will still do on their task. KL Divergence measures the difference in two probability distributions, which in this case are the merged model and the fine tuned models, which is a better measure of loss during merging compared to other loss functions because of how it directly measures how different the models end up being.

Additionally, the algorithm uses Sharpness-Aware-Minimization in order to find minima that are more flattened rather than a minima at a very sharp curve. This is because minima that are more flattened are more generalized and less prone to drastic changes upon slightly tuning the model weights, as opposed to sharp minima which may be overfitting.

# Results And Evaluation

TBA

# Novelty and Conclusion

The novelty in this paper is providng a new way to merge models that drastically reduces the amount of data needed for model merging while maintaing higher accuracy and speed than 





