# Final Project

We plan on studying the ability of LSTMs to understand the licensing of negative
polarity items in English. Negative polarity items are a class of words like
“ever”, “any”, and “even” which are only allowed to occur in certain “negative”
contexts.

1. He never gave up.
2. *He ever gave up.
3. Nobody ever gave up.

Notably, the relationship between the licensor and the negative polarity item is
not based solely on linear word order, but rather on the hierarchical structure
of the sentence. In particular, the licensor must c-command the negative
polarity item. Thus, if a language model can correctly predict the distribution
of negative polarity items, it suggests that the model has learned certain
aspects of syntactic structure.

In order to assess an LSTM’s accuracy in predicting the distribution of negative
polarity items, we could use the surprisal metric described in Wilcox et al.
(2018).

S(xi)=-log2 p(xi|hi-1)

A high surprisal metric means that a given word was unexpected in its context.
In order to probe the model’s understanding of negative polarity licensing, we
could construct a synthetic dataset which contained both grammatical examples,
in which negative polarity items are correctly licensed, and ungrammatical
examples, in which the licensor is either not present or is not in a c-command
relationship with the licensee.

If the model has learned the rules governing negative polarity licensing, we
should expect to see a high surprisal in the ungrammatical examples, and a
relatively lower surprisal in the grammatical examples.

Next, in order to gain a deeper understanding into how the model’s knowledge is
stored, we could attempt to use an ablation strategy similar to that employed in
Lakretz et al. (2019). By systematically ablating (disabling) units in the
LSTM’s cell state, we can observe which units have the greatest impact on the
LSTM’s success. If a small number of units have a very significant effect on the
model’s success, it would suggest that the model’s knowledge is stored locally.
Otherwise, we could infer that the knowledge is distributed throughout the
network.

If we do find that the model’s knowledge is stored locally, then we could
continue to perform a deeper analysis of how exactly the input gate, forget
gate, output gate, candidate cell state, and cell state change over time, as a
sentence is parsed. This analysis would be similar to that in Lakretz et al.
(2019).

I imagine that the focus of this project will be on studying an existing LSTM
rather than training our own model. The Google model from Jozefowicz et al.
(2016) and the Gulordava model from Gulordava et al. (2018) have both been used
in similar experiments, and we could probably focus our project on either one of
these, or on both.

The methodology described above is inspired closely by both Wilcox et al. (2018)
and Lakretz et al. (2019).
