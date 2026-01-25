## Background information

I want to build a prompt as such a way for the training stage:

1. Ask LLM to generate a model
2. Give the baseline model as the reference model.
3. Ask LLM to generate the vision model code that improves the accuracy beyond the baseline model.

For example:
```
"## Role",
"You are a visionary deep learning architect renowned for designing breakthrough neural networks by drawing inspiration from diverse scientific domains.",
"## Task",
"Create an innovative vision model that maximizes '{metric}' on '{dataset}' for '{task}' at epoch {epoch}. Reference model achieved {accuracy} — use it as inspiration, NOT a constraint.",
...
"",
"Reference configuration: ",
"<hp>{prm}</hp>",
"<tr>{transform_code}</tr>",
"<metric>{metric_code}</metric>",
"<nn>{nn_code}</nn>",
"",
"## Limitations",
...
```

The output is:
```
<hp>{addon_prm}</hp>
<tr>{addon_transform_code}</tr>
<nn>{addon_nn_code}</nn>
```

I want to ask the language model to generate the improved code of vision model based on the same condition as the reference model(metric, dataset, task, epoch).

## Problem

But in nn-dup the function `curate_from_lemur` used to build the data only keep data of a single row, so I need to modify the function to keep additional one row for the same condition as the reference model but with an improved accuracy.

### Steps:

Tip: 
- This is the Possible steps, you can change the steps if you think it is not correct.
- You can use git in nn-dup to look which files you need to change.

1. For each data rows, find another one additional model with an improved accuracy and with the same conditions(metric, dataset, task, epoch) but with a different nn code.
2. Modify the prompt_builder.py and any other files that related to the prompt building to handle this updating.

## Expected result

After modification, The output should be the model code that having higher accuracy than the reference model.