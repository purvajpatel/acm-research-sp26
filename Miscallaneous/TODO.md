Finished:

1. Find the 5 models to use. - should use the same model. As for which model, it will be bert-base-uncased. If I have time, I'd like to try doing the fine tuning portions myself.
2. Find the 5 tasks each model will do, and find labeled data for those. Since we are assuming already fine-tuned models, we don't have to do this. Again, with time I'd like to do the fine tuning myself.
3. Fine tune (or find the tuned weights for) the models with the labeled data for that task, get the fine tuned weights. See above.
4. Get ~32 unlabeled datapoints for training per task, 8 unlabeled datapoints for validation per task, and 20 unlabeled datapoints for testing per task. Try to find these instead of generating them. Done.

Remaining:
5. Look at the Github and how they implement the model that optimizes with Sharpness Awareness Optimization (SAM) and KL Divergence, and implement that algorithm.
6. Evaluate accuracy of merged model on each task and its average accuracy. Additionally, report average runtime for each task as well.
7. Write the final report.

Bonus:
8. If you have time, try the fine tuning yourself, maybe with a different model.