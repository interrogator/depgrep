# depgrep

Dependency parse searching for CONLL-U DataFrames

<!--- Don't edit the version line below manually. Let bump2version do it for you. -->
> Version 0.1.2

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
results = corpus.depgrep(query)
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

If your corpus has metadata fields, you can also incorporate these: `speaker"TONY"` will match tokens with TONY given as speaker. Since such metadata is given at sentence level, however, it is probably better to first reduce your DataFrame to sentences with the correct metadata.

### Relations

Relations specify the relationship between nodes. For example, we can use `f"nsubj" <- f"ROOT"` to locate nominal subjects governed by nodes in the role of *ROOT*. The thing you want to find is the leftmost node in the query. So, while the above query finds nominal subject tokens, you could use inverse relation, `f"ROOT" -> f"nsubj"` to return the ROOT tokens.

Available relations:

```
a = b : a and b are the same node
a & b : a and b are the same node (same as =)

a <- b : a is a dependent of b
a <<- b : a is a descendent of b, with any distance in between
a <-: b : a is the only dependent of b
a <-N b : a is descendeent of b by N generations

a -> b : a is the governor of a
a ->> b : a is an ancestor of b, with any distance in between
a ->: b : a is the only governor  of b (as is normal in many grammars)
a ->N b : a is ancestor of b by N generations

a + b : a is immediately to the left of b
a ++ b : a is two places to the left of b
a +++ b : a is three places to the left of b (...)
a +N b : a is N places to the left of b
a <| b : a is left of b, with any distance in between

a - b : a is immediately to the right of b
a -- b : a is two places to the right of b
a --- b : a is three places to the right of b
a -N b: a is n places to the right of b
a |> b : a is right of b, with any distance in between

a $ b : a and b share a governor
```

### Negation

Add `!` before a relation to negate it: `f"ROOT" != x"VERB"` will find non-verbal ROOT nodes.

### Brackets

Brackets can be used to make more complex queries:

```
(f"amod" = l/^[abc]/) <- (f/nsubj/ != x/NOUN/)
```

translates to *match adjectival modifiers starting with a, b or c, which are governed by nominal subjects that are not nouns*
