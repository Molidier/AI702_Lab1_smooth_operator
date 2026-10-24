!!! I DELETED ALL THE QUESTIONS IN THIS FILE AS WAS SAID DURING THE LAB SESSION AND ANSWERED TO THE QUESTIONS FROM THE IN LAB

permute vs reshape:
every value is moved together with the dimensions when they are permuted. the order of the indices has changed, but the value for picture n, channel c, row h, and column w remains unchanged.
nothing is moved by reshaping. it divides the values into the new shape in the order that they are stored. the values end up in the incorrect places even though the shape appears correct.
for instance, the three channel values for the first pixel in the first image are 0, 20, and 40. permute yields 0, 20, and 40, which is accurate. it is incorrect since reshape returns 0, 1, and 2, which are simply the subsequent values in memory from the same channel.
memory: .contiguous() creates a fresh copy in the correct order. after permute, the values are no longer stored in order (the tensor is not contiguous). this caused view to fail on the transposed tensor in my run, but reshape succeeded.

### My run ID's
baseline run: 20261024T061820-aadc7823, in lab0/results/baseline/
main (core) run i submitted: 20261024T093223-4928d213, in lab0/results/core/. i also did earlier core runs (20261024T062346-a1d29cd4 and 20261024T080539-0047f45f), but i did not submit them.
transfer run: 20261024T093808-34773ffb, in lab0/results/transfer/. i also did an earlier transfer run (20261024T082730-04e3d000), but i did not submit it.
files i used for my answers: validation_predictions.csv, metrics.json, details.json, raw.npz and settings_used.json
wrong predictions: in the core run, samples 40 and 263; in the transfer run, samples 135, 414 and 128.

### accuracy audit (core run 20261024T093223-4928d213):
reported_accuracy = 97.80 with units "fraction between 0 and 1" recalculated from all 91 csv rows: 89/91 = 0.978 (97.80%) The unit label is incorrect because 97.80 is a percentage rather than a fraction.

Problem 3 Additional experiment

transfer setting: seed 711 instead of 702 (settings_used.json). it changes the data split and the model starts from different initial weights.

expectation: most probably validation set results will be different, because train set and validation set will contain different sets of data. that also affects the weights as i stated before. it can affect the validation set accuracy and the misclassified IDs. but the number of data in both sets should remain the same or nearly the same.

comparison (transfer run 20261024T093808-34773ffb):
accuracy recomputed from the csv: 88/91 = 0.967 (96.70%)
core was 89/91 = 0.978 (97.80%)
we can see that the transfer run has one more mistake. my assumption was correct and the misclassified IDs changed from 40 and 263 to 135, 414 and 128.
reported_accuracy = 96.70, still labelled as "fraction between 0 and 1".
the unit problem still stayed because the seed does not affect the way accuracy was calculated.

help received: Rozalina explained the structure of the lab. documentations of the numpy and pythorch.