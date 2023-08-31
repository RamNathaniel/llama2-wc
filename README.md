# Word completion for LLMs
The goal is to speed up LLM inference by taking shortcuts from running the entire
LLM for every token - only run it for the tokens that we need it for.

Impact should be larger the larger the LLM is.

## Insights

- At some positions, LLaMA is absolutely sure (100%) that there is a suffix.
- These are probably for puctuations (need to verify)
- Detecting those is probably easy and does not require a long context window - just the current sentense.

> First attempt should be:
Create a classifier for those and decide which suffix to put there.
Go for the cases of > 95% of puctuations. Maybe even 100%.

## Future directions

### Single puctuations only

Focus only on adding a puctuation, but do it reliably almost as LLaMA does it
so you save up those instances.

- How much are we actually saving? What's the frequency of 100% certainty of puctuation?
- How easy is it to detect such a thing?

### Word completion

Complete all words that have a suffix

#### Cases seen so far

- Puctuations
- Words that don't really live by themselves (more rare)
- Words that can live by themselves, but not in this context.

#### Open questions

- Do we need more info from previous text? Or would simple LLM model (short context) would do?

### Sentance phrasing by small model

Hypothesis:
The LLM knows what it is going to write in the entire sentence buy is only used to predict the next token. We can use the information inside it to phrase in a more lightweight model.

TODO:
Get information from deeper in the LLaMA network, and try to guess the entire sentance

#### Open questions

- Will the writing style be hurt?
- Does the LLM have enough data to phrase the entire sentance or just the next token?
- How do we know what is a sentance? When do we stop predicting with the small model?
- Can we have a running confidence?

# Technicalities

## Code
- Currently using 7B, using the pyllama base repo.
- textfile_gen.py - generator of tokens from text file.
- inference.py - run the base model (LLaMA) and return the probabilities (single GPU)
- ram-tokenizer.py - utility to get tokenizer for entered text + list of suffix tokens
- wc_train.py - train wc_model with the outputs of LLaMA (on single GPU, batch = 1)

## Folders
- ~/checkpoints - where to save models
- ~/lib - where to save text corpus to train on (https://pypi.org/project/epub2txt/, books from https://www.planetebook.com/)
```bash
wget ...
epub2txt -f book.epub
```
- ~/llama/llama-2-7b - where the weights are
- ~/llama/tokenizer.model, tokenizer.checklist.chk - tokenizer model used
