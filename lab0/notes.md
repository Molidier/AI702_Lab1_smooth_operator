Question one:
initial: (6x4)
two features: (6x2)
weight: (3x2)
Two features and three scores per example: (6x2)(2x3) = (6x3)
b is (3,) -> one number per output score
Final XW^T + b -> (6x2)(2x3) + (3,) = (6x3)

Batch of one:
(1x4)->(1x2)
(1x2)(2x3) + (3,) = (1x3)

As we can see W and b keep the same shapes. It means they only depend on the number of features and outputs

Explain why preserving the leading singleton dimension (the batch 
dimension) is important when the batch size is one:
It preserves the rule that rows are examples and columns are outputs so the same code works for one example or many.

(b) Validation examples must be standardized using statistics estimated from the training split. Explain 
what changes if statistics are instead recomputed from the validation examples. Specify a check that would 
detect accidental mutation of the original data during standardization. 

1. Validation influences preprocessing -> data leakage
2. We change the reference -> it hides the shift of the validation set relative to training set
3. example depends from others that are in the same group with it giving each time different result.

to detect mutation we need to save a copy of the data before standardization and compare it with the original after the function runs. if they are not equal the function changed the original data. we can also check that the result is a new array rather than x's memory.

(c) Consider y = [[0], [1], [1], [0]] and p = [0, 1, 0, 0], represented as NumPy arrays. Determine the shape and value of 
(y == p).mean(). Compare the corresponding entries in the two arrays and compute the rate of matching entries. 
Explain why checking only that a reported accuracy lies between zero and one would miss this error.  

y: (4x1), p: (4,)
broadcasting -> (4x1) vs (1x4) -> (4x4)
each label is compared with every prediction

y == p =
[[1 0 1 1]
[0 1 0 0]
[0 1 0 0]
[1 0 1 1]]

row sums: 3, 1, 1, 3 -> 8 matches out of 16
(y == p).mean() = 8/16 = 0.5

correct comparison, entry by entry:
y_0 = 0, p_0 = 0 -> match
y_1 = 1, p_1 = 1 -> match
y_2 = 1, p_2 = 0 -> no match
y_3 = 0, p_3 = 0 -> match
-> 3/4 = 0.75

so the broadcast version gives 0.5 instead of the true 0.75.

checking only that accuracy is between 0 and 1 would not catch this error, because 0.5 is still a valid fraction.
the problem is not the range of the value but which entries were compared. every label was compared with every prediction instead of row by row.
to detect it we need to check that y and p have the same shape before comparing them or recompute accuracy from matched rows.

Problem 2  Reproducibility 
(a) Two experiment reports show the same rounded accuracy. What would you need to check to determine 
whether they used the same examples, predictions, settings and code? Explain why an accuracy value or a code 
version alone is not enough. 

same accuracy doesn't mean same results, because it does not identify the examples, predictions or procedure.
code version also doesn't guarantee reproducibility -> same code can run with different data settings, random seeds or model weights and produce different results.
that's why we need to chech are this things the same: examples, predictions, settings, code and environment.

(b) A reported accuracy is numerically plausible but has no stated unit or record of how rows were matched. 
Propose two independent checks that could establish whether it is valid. Explain which type of error each check 
would detect.

1. we need to recompute accurace and verify its unit. it can be given as fraction or percentage. it detects unit and calculation errors.
2. we need to verify that predictions are matched to the correct examples. make sure that row positions correspond to the true labels, or match predictions and true labels using unique example IDs. it detects row alignment and sample selection errors such as wrong, missing or duplicated rows.

credits: numpy and pythorch documentaions