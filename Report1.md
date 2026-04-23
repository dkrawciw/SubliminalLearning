## Questions we are answering

* Can we use prompt engineering rather than fine tuning on the student model?

* Can we find a way to generalize this behavior so it works between different models, rather than instances of the same model?

## Current Experiments

### Mnist Classifier

We are currently working on one of the minor results from the paper, where they create an Mnist classifier with 13 logits, so only 10 of them will correspond to numbers, and training second one only on replicating the 3 unrelated logits, which results in a decent Mnist classifier despite no seemingly relevant training data. 

## Planned Experiments

### Experiment 1: Replicating Results

First we want to replicate the results of the paper, which will involve first fine tuning a “teacher model” for misalignment, then having it generate seemingly random/unrelated text, and then fine tuning a second instance of the same model, the “student model” on that text, and measuring misalignment through a series of specific prompts.

### Experiment 2: Prompt Engineering

Next, if these results do indeed hold, we would like to experiment with prompt engineering rather than fine tuning for the student model. This will involve inputting the generated text from the parent into the student model along with our actual test prompts, and seeing if we can obtain similar results to fine tuning. The current hypothesis is that if this worked well, the subliminal learning paper would have already tried it and published those results as well, so we don’t expect very astounding results with this theory.

### Experiment 3: Communication Between Models

This experiment holds promise because it aligns with the mathematical theory given in section 6 of the subliminal learning theory. Specifically, after 1 step of gradient descent on any outputs from a teacher model, the loss from a student model on a dataset will not move further from the loss of the teachers on the same set (so it will likely be closer to the teachers). This theory should theoretically work between different models and across multiple training sets, so fine tuning a model with lots of data from another should bring their outputs closer, which we hope will result in similar “subliminal messaging code”, despite different model structure. 
	
Additionally, if that doesn’t end up working, this could be attempted with a model with the exact same matrix structure, just no guarantee as to the internal weights, which feels much more likely to succeed.

A problem that could arise from this is a similar problem mentioned in the paper, that the models are trained using stochastic gradient descent, so this theory isn’t guaranteed always, only probabilistically. Since the results seem to hold in the paper's results despite this, it should hold here as well.

## How The Answers Change Depending On The Results

If experiment 1 fails, then we cannot ensure that the results of this paper do indeed hold, so our project will be ill-posed, and we will shift to attempting to explain why we were not able to replicate the results, or why they are not guaranteed to work. We think this is very unlikely as this paper comes from a very reputable source.

If experiment 2 fails, which is reasonably likely, then we can say it is unlikely that prompt engineering, at least in the way we end up implementing it, cannot transmit subliminal messages in the same way as with finetuning, and we will try to find an explanation as to why. Additionally, for experiments on more complex models that we may have to pay for, it will cost more money to fine tune the model.

If experiment 3 fails, then the statements of the authors of the subliminal learning paper will hold even more weight, as there is no guarantee that the hidden encoding can work between models with different structure.

## Roadblocks