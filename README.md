# 1. Language-Conditioned Graph Networks for Relational Reasoning

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