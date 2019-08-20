# depgrep

Dependency parse searching for CONLL-U DataFrames

<!--- Don't edit the version line below manually. Let bump2version do it for you. -->
> Version 0.1.3

> Note: this tool currently doesn't have tests, CI, etc. It is not yet advised to use this tool outside of the depgrep methods provided by the `buzz` library.

## Installation

```bash
pip install depgrep
```

## Usage

The tool is designed to work with corpora made from CONLL-U files and parsed into DataFrames by *buzz*. The best thing to do is use *buzz* to model corpora, and then use its *depgrep* method.

```bash
pip install buzz
```

Then, in Python:

```python
from buzz import Corpus
corpus = Corpus('path/to/conll/files')
query = 'l"have"'  # match the lemma "have"
```

## Syntax

depgrep searches work through a combination of *nodes* and *relations*, just like Tgrep2, on which this tool is based.

### Nodes

A node targets one token feature (word, lemma, POS, wordclass, dependency role, etc). It may be specified as a regular expression or a simple string match: `f/amod|nsubj/` will match tokens filling the *nsubj* or *amod* role; `l"be"` will match the lemma, *be*.

The first part of the node query chooses which token attribute is to be searched. It can be any of:

```
w : word
l : lemma
p : part of speech tag
x : wordclass / XPOS
f : dependency role
i : index in sentence
s : sentence number
```

Case sensitivity is controlled by the case of the attribute you are searching: `p/VB/` is case-insensitive, and `P/VB/` is case sensitive. Therefore, the following query matches words ending in *ing*, *ING*, *Ing*, etc:

```
w/ing$/
```

For case-insensitivity across the query, use the `case_sensitive=False` keyword argument.

### Relations

Relations specify the relationship between nodes. For example, we can use `f"nsubj" <- f"ROOT"` to locate nominal subjects governed by nodes in the role of *ROOT*. The thing you want to find is the leftmost node in the query. So, while the above query finds nominal subject tokens, you could use inverse relation, `f"ROOT" -> f"nsubj"` to return the ROOT tokens.

Available relations:

```
a = b   : a and b are the same node
a & b   : a and b are the same node (same as =)

a <- b  : a is a dependent of b
a <<- b : a is a descendent of b, with any distance in between
a <-: b : a is the only dependent of b
a <-N b : a is descendent of b by N generations

a -> b  : a is the governor of a
a ->> b : a is an ancestor of b, with any distance in between
a ->: b : a is the only governor of b (as is normal in many grammars)
a ->N b : a is ancestor of b by N generations

a + b   : a is immediately to the left of b
a +N b  : a is N places to the left of b
a <| b  : a is left of b, with any distance in between

a - b   : a is immediately to the right of b
a -N b  : a is n places to the right of b
a |> b  : a is right of b, with any distance in between

a $ b   : a and b share a governor (i.e. are sisters)

a $> b  : a is a sister of and to the right of b.
a $< b  : a is a sister of and to the left of b.

```

### Negation

Add `!` before a relation to negate it: `f"ROOT" != x"VERB"` will find non-verbal ROOT nodes.

### Brackets

Brackets can be used to make more complex queries:

```
f"amod" = l/^[abc]/ <- (f/nsubj/ != x/NOUN/)
```

The above translates to *match adjectival modifiers starting with a, b or c, which are governed by nominal subjects that are not nouns*

Note that **without** brackets, each relation/node refers to the leftmost node. In the following, the plural noun must be the same node as the *nsubj*, not the *ROOT*:

```
f"nsubj" <- f"ROOT" = p"NNS"
```

### *Or* expressions

You can use the pipe (`|`) to create an *OR* expression.

```
# match all kinds of modifiers
x"ADJ" | f"amod" | f"appos" | p/^JJ/
x"NOUN" <- f"ROOT" | = p"NNS"
```


Above, we match nouns that are either governed by *ROOT*, or are plural.

### Wildcard

You can use `__` or `*` to stand in for any token. To match any token that is the governor of a verb, do:

```
__ -> x"VERB"
```
