# Set-Up Instructions
1. Install the required packages in requirements.txt. You can simply open the terminal and run 'pip install -r requirements.txt'
2. Download the CLEVR dataset. My implementation expects all downloaded data folders to be in a directory called 'data'. That is, you should have data > CLEVR_v1.0. Download the CLEVR dataset: https://cs.stanford.edu/people/jcjohns/clevr/. Scroll down to 'Download' and click '
Download CLEVR v1.0 (18 GB)'.

# 1. Language-Conditioned Graph Networks for Relational Reasoning
[Paper](https://arxiv.org/abs/1905.04405)

## Overview
Visual question answering (VQA) and referring expression comprehension (REF) are grounded comprehension tasks that require models to reason about entity relationships in the context of the task. Language-Conditioned Graph Networks (LCGN) are graphical frameworks that represent entities as nodes, which are described by context-aware representations, or edges, from related nodes using iterative message parsing.

## Motivation
Previous works have attempted to learn joint visual-text representations, pool over pair-wise relationships, or build explicit reasoning steps to identify relationships between objects. However, these methods are unable to capture contextual details about the objects, such as position (ie, an object being under a table vs on top of a table), and alternatively rely on complex models or inference rules to answer questions. This makes it difficult to repurpose these techniques for new tasks.

## Novelty
The LCGN model is a graph network that dynamically builds relational edges between objects (nodes). It conditions the message passing on specific contextual relationships described in the text rather than building a representation of all n-order relations for each object, making the technique efficient and broadly applicable.

## Advantages/Disadvantages
### Advantages
* Dynamically learns relationships between objects rather than just object features
* Graph edges can give insight into how the model learns which objects are relevant based on the specific question, showing the importance of the relationship
* Can work with differing number of objects and questions without changing the graph structure manually
### Disadvantages
* LCGN doesn't inherently learn relationships, but uses language to determine graph structure
* Doesn't transfer well to ambiguous questions or previously unseen types of questions
* Relies on annotated scene graphs (like the ones provided in the CLEVR dataset) and may see a drop in performance with un-annotated data

## Implementation
I implemented a simple version of LCGN using the CLEVR dataset, which contains natural language questions that are answered by functional programs describing the reasoning steps necessary to answer the question based on an image. It also includes scene graphs, providing the ground-truth positions of objects, attributes, and object relationships. In my Language-Conditioned Graph Network, each image is represented as a set of object nodes with features for shape, color, material, and size. The question encoder converts the question into a vector embedding, which conditions the graph. The graph network then computes edges between objects based on the question, allowing message passing along these learned relationships to aggregate information. Lastly, a classifier predicts the answer to the question using the updated object representations generated using message passing.

# 2. Deep Compositional Question Answering with Neural Module Networks
[Paper](https://arxiv.org/abs/1511.02799)

## Overview
Visual question answering (VQA) is inherently compositional, meaning questions can be broken into subtasks. This paper seeks to build and jointly-train neural module networks (NMN) to construct deep networks for question answering (QA). The modular networks are instantiated on linguistic substructures of questions, and they have reusable components to carry out specific tasks. They are then jointly trained for QA and composed to form deep networks.

## Motivation
This paper is motivated by the absence of a single 'best' system for VQA. However, it is common for vision systems to be initialized with a "prefix of a network trained for classification" (that is, learnable tokens/embeddings that are added to an input prior to model processing), which reduce training time and increase accuracy. Thus, networks are not universal, but they are modular, and the intermediate representations of one task can be re-used in others.

## Novelty
This paper seeks to generalize the reusability of NMNs to construct modular networks from the outputs of a semantic parser used to separate a question into subtasks and use them to complete VQA tasks. Traditional approaches to QA view the question as a single problem with the goal of learning a single function to map to an answer. This paper views complex problems as a "highly-multitask learning setting" (that is, in order to solve the problem, one must first solve multiple subproblems, which are technically unbounded and may themselves consist of even smaller subproblems).

## Advantages/Disadvantages
### Advantages
* Improved performance on complex questions
* Generalizes to more complex questions that were unseen during training
### Disadvantages
* Answers for yes/no questions may suffer from overfitting in the measure module
* Predictions suffer from semantic confusion, lexical variation, and plausible but unrelated-to-image responses
* Parser tends to pick up irrelevant details for complex questions (may be fixed with joint learning)

## Implementation
I implemented a minimal Neural Module Network (NMN) demo using the CLEVR dataset, where each question is paired with a functional program that describes the reasoning steps needed to answer it from an image. I built a small module library including attend, relate, combine, classify, unique, and measure operations, and converted CLEVR programs into a simplified NMN format so the composer can execute them sequentially. Images are passed through a frozen ResNet backbone to produce spatial feature maps, and the NMN composer interprets each program token to run the appropriate module on the feature maps or attention maps, producing an answer distribution. The model is trained end-to-end only through the module parameters (not the backbone), and the final answer is predicted by the measure module after executing the program.