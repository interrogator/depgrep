#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# Natural Language Toolkit: depgrep search
#
# Copyright (C) 2001-2019 NLTK Project
# Author: Will Roberts <wildwilhelm@gmail.com>
# URL: <http://nltk.org/>
# For license information, see LICENSE.TXT

"""
depgrep, modified from nltk tgrep
"""

import functools
import re

from six import binary_type, text_type

import pyparsing
import nltk.tree


class DepgrepException(Exception):
    """
    depgrep exception type.
    """
    pass


def _np_attr(node, pos, cs):
    if isinstance(node, str):
        return node if cs else node.lower()
    return node[pos] if cs else node[pos].lower()


def _ancestors(node, values):
    """
    Returns the list of all nodes dominating the given tree node.
    This method will not work with leaf nodes, since there is no way
    to recover the parent.
    """
    _ancestors = []
    gov = node['g']
    sent = values.xs((node.name[0], node.name[1]), level=["f", "s"])
    while gov:
        gov_tok = sent.loc[gov]
        _ancestors.append(gov_tok)
        gov = gov_tok['g']
    return _ancestorst.parent()
    return results


def _descendants(node, values):
    """
    Returns the list of all nodes which are descended from the given
    tree node in some way.
    """
    raise ValueError('Broken')
    sent = values.xs((node.name[0], node.name[1]), level=["f", "s"])
    # a bag of _governors
    govs = {node.name[2]}
    size = len(govs)

    while True:
        # reduce sent to just _dependents
        children = sent[sent['g'].isin(govs)]
        # add these dependent ids to possible _governors
        govs = govs.union(set(children.index))
        if size == len(govs):
            break
        size = len(govs)
    govs.remove(node.name[2])
    return [sent.loc[i] for i in govs]


def _before(node, values, places=False):
    """
    Returns the set of all nodes that are before the given node.
    """
    f, s, i = node.name
    matches = values.loc[f, s, slice(None, i - 1)]
    aslist = [matches.loc[i] for i in matches.index]
    if places is not False:
        places = -places
        aslist = [aslist[places]]
    return aslist


def _immediately_before(node, values, positions):
    """
    Returns the set of all nodes that are immediately after the given
    node.

    Tree node A immediately follows node B if the first terminal
    symbol (word) produced by A immediately follows the last
    terminal symbol produced by B.
    """
    if isinstance(node, pd.Series):
        i = node._n - 1
    else:
        i = node[positions['_n']] - 1
    try:
        return [values.iloc[i]]
    except:
        try:
            return [values[i]]
        except:
            return list()


def _after(node, values, places=False):
    """
    Returns the set of all nodes that are after the given node.
    """
    f, s, i = node.name
    matches = values.loc[f, s, slice(i + 1, None)]
    aslist = [matches.loc[i] for i in matches.index]
    if places is not False:
        aslist = [aslist[places - 1]]
    return aslist


def _immediately_after(node, values, positions):
    """
    Returns the set of all nodes that are immediately after the given
    node.

    Tree node A immediately follows node B if the first terminal
    symbol (word) produced by A immediately follows the last
    terminal symbol produced by B.
    """
    if isinstance(node, pd.Series):
        i = node._n + 1
    else:
        i = node[positions['_n']] + 1

    try:
        return [values.iloc[i]]
    except:
        try:
            return [values[i]]
        except:
            return []


def _depgrep_macro_use_action(_s, _l, tokens):
    """
    Builds a lambda function which looks up the macro name used.
    """
    assert len(tokens) == 1
    assert tokens[0][0] == '@'
    macro_name = tokens[0][1:]

    def macro_use(n, m=None, el=None):
        if m is None or macro_name not in m:
            raise DepgrepException('macro {0} not defined'.format(macro_name))
        return m[macro_name](n, m, el)

    return macro_use


def _depgrep_node_action(_s, _l, tokens, positions, case_sensitive=False):
    """
    Builds a lambda function representing a predicate on a tree node
    depending on the name of its node.
    """
    # print 'node tokens: ', tokens
    if tokens[0] == "'":
        # strip initial apostrophe (depgrep2 print command)
        tokens = tokens[1:]
    if len(tokens) > 1:
        # disjunctive definition of a node name
        assert list(set(tokens[1::2])) == ['|']
        # recursively call self to interpret each node name definition
        tokens = [
            _depgrep_node_action(None, None, [node], positions, case_sensitive=case_sensitive)
            for node in tokens[::2]
        ]
        # capture tokens and return the disjunction
        return (lambda t: lambda n, m=None, el=None: any(f(n, m, el) for f in t))(tokens)
    else:
        if hasattr(tokens[0], '__call__'):
            # this is a previously interpreted parenthetical node
            # definition (lambda function)
            return tokens[0]

        # determine the attribute we want to search (word, lemma, pos, etc)
        if tokens[0][0].lower() in list('siwlxpmgfeo') and not tokens[0].startswith('i@'):
            attr = tokens[0][0].lower()
            tokens[0] = tokens[0][1:].lower()
        else:
            attr = 'w'

        pos = positions[attr]

        # if the token is 'anything', just return true
        if tokens[0] == '*' or tokens[0] == '__':
            return lambda n, m=None, el=None: True
        # if it's a quote, it must be an exact match
        elif tokens[0].startswith('"'):
            assert tokens[0].endswith('"')
            node_lit = tokens[0][1:-1].replace('\\"', '"').replace('\\\\', '\\')
            if not case_sensitive:
                node_lit = node_lit.lower()
            node_lit = node_lit.split(',')
            return (
                lambda s: lambda n, m=None, el=None: any(
                    _np_attr(n, pos, cs=case_sensitive) == x for x in s
                )
            )(node_lit)

        # if it's slashes, it's a regex
        elif tokens[0].startswith('/'):
            assert tokens[0].endswith('/')
            node_lit = tokens[0][1:-1]
            if not case_sensitive:
                return (
                    lambda r: lambda n, m=None, el=None: r.search(
                        _np_attr(n, pos, cs=case_sensitive)
                    )
                )(re.compile(node_lit, re.IGNORECASE))
            else:
                return (
                    lambda r: lambda n, m=None, el=None: r.search(
                        _np_attr(n, pos, cs=case_sensitive)
                    )
                )(re.compile(node_lit))

        elif tokens[0].startswith('i@'):
            node_func = _depgrep_node_action(
                _s, _l, [tokens[0][2:].lower()], positions, case_sensitive=case_sensitive
            )
            return (
                lambda f: lambda n, m=None, el=None: f(_np_attr(n, pos, cs=case_sensitive).lower())
            )(node_func)
        else:
            return (lambda s: lambda n, m=None, el=None: _np_attr(n, pos, cs=case_sensitive) == s)(
                tokens[0]
            )


def _depgrep_parens_action(_s, _l, tokens):
    """
    Builds a lambda function representing a predicate on a tree node
    from a parenthetical notation.
    """
    # print 'parenthetical tokens: ', tokens
    assert len(tokens) == 3
    assert tokens[0] == '('
    assert tokens[2] == ')'
    return tokens[1]


def _depgrep_nltk_tree_pos_action(_s, _l, tokens):
    """
    Builds a lambda function representing a predicate on a tree node
    which returns true if the node is located at a specific tree
    position.
    """
    # recover the tuple from the parsed sting
    node_tree_position = tuple(int(x) for x in tokens if x.isdigit())
    # capture the node's tree position
    return (
        lambda i: lambda n, m=None, el=None: (
            hasattr(n, 'treeposition') and n.treeposition() == i
        )
    )(node_tree_position)


def _governor(row, values, positions):
    """
    Get governor of row as row from values
    """
    if not row[positions['g']]:
        return 'ROOT'
    else:
        i = row[positions['i']]
        n = row[positions['_n']]
        return [values[n - int(i) + row[positions['g']]]]


def _sentence(row, values, positions):
    slen = row[positions['sent_len']]
    rn = row[positions['_n']]
    i = row[positions['i']]
    start = rn - i + 1
    end = start + slen
    return values[start:end]


def _dependents(row, values, positions):
    """
    Get list of dependents of row as rows from values
    """
    sent = _sentence(row, values, positions)
    i = row[positions['i']]
    try:
        return sent[sent[:, positions['g']] == i]
    except:
        return sent[sent[:, positions['g']] == row[positions['i']]]


def _governors(row, values, positions):
    """
    Sometimes we need a list of _governors, but only one is possible
    So, here we just return a one-item list
    """
    return [_governor(row, values, positions)]


def _has_dependent(row, values, positions):
    """
    Check if dependents exist

    Return: bool
    """
    sent = _sentence(row, values, positions)
    return sent[sent[positions['g']] == row[positions['i']]].any()


def _has_governor(row, positions):
    """
    Check if governor exists

    Return: bool
    """
    return row[positions['g']] not in ['_', '0', 0]


def _sisters(row, values, positions):
    """
    Give a list of tokens sharing a governor, but not row
    """
    sent = _sentence(row, values, positions)
    sent = sent[sent[positions['g']] == row[positions['g']]]
    # almost definitely broken
    return sent.drop(getattr(row, 'name', tuple(row[:3])), errors='ignore')


def _depgrep_relation_action(_s, _l, tokens, values, positions):
    """
    Builds a lambda function reprevaluesing a predicate on a tree node
    depending on its relation to other nodes in the tree.
    """
    # print 'relation tokens: ', tokens
    # process negation first if needed
    negated = False
    if tokens[0] == '!':
        negated = True
        tokens = tokens[1:]
    if tokens[0] == '[':
        # process square-bracketed relation expressions
        assert len(tokens) == 3
        assert tokens[2] == ']'
        retval = tokens[1]
    else:
        # process operator-node relation expressions
        operator, predicate = tokens

        # A = B      A and B are the same node
        if operator == '=' or operator == '&':
            retval = lambda n, m=None, el=None: (predicate(n, m, el))
        # A -> B       A governs B.
        elif operator == '->':
            retval = lambda n, m=None, el=None: (
                any(predicate(x, m, el) for x in _dependents(n, values, positions))
            )
        # A <- B       A is a dependent of B.
        elif operator == '<-':
            retval = lambda n, m=None, el=None: (any(predicate(x, m, el) for x in _governor(n, values, positions)))

        # A ->> B      A dominates B (A is an ancestor of B).
        elif operator == '->>':
            retval = lambda n, m=None, el=None: (
                any(predicate(x, m, el) for x in _descendants(n, values, positions))
            )
        # A <<- B      A is dominated by B (A is a descendant of B).
        elif operator == '<<-':
            retval = lambda n, m=None, el=None: any(predicate(x, m, el) for x in _ancestors(n, values, positions))

        # A ->: B      B is the only child of A
        elif operator == '->:':
            retval = lambda n, m=None, el=None: (
                len(_dependents(n, values, positions)) == 1 and predicate(_dependents(n, values, positions)[0], m, el)
            )

        # A <-: B      A is the only child of B.
        elif operator == '<-:':
            retval = lambda n, m=None, el=None: (
                len(_governors(n, values, positions)) == 1 and predicate(_governor(n, values, positions), m, el)
            )
        # A ->N B      B is the Nth child of A (the first child is <1).
        elif operator[:2] == '->' and operator[2:].isdigit():
            idx = int(operator[1:])
            # capture the index parameter
            retval = (
                lambda i: lambda n, m=None, el=None: (
                    0 <= i < len(_dependents(n, values, positions)) and predicate(n[i], m, el)
                )
            )(idx - 1)
        # A <-N B      A is the Nth child of B (the first child is >1).
        elif operator[:2] == '<-' and operator[2:].isdigit():
            idx = int(operator[1:])
            # capture the index parameter
            retval = (
                lambda i: lambda n, m=None, el=None: (
                    0 <= i < len(_governors(n, values, positions))
                    and (n is _governors(n, values, positions)[i])
                    and predicate(_governors(n, values, positions), m, el)
                )
            )(idx - 1)

        # the next four are going to be pretty rare...
        # A ->- B      B is the last child of A (synonymous with A <-1 B).
        elif operator == '->-' or operator == '->-1':
            retval = lambda n, m=None, el=None: (
                _has_dependent(n, values, positions) and predicate(_dependents(n, values, positions)[-1], m, el)
            )
        # A -<- B      A is the last child of B (synonymous with A >-1 B).
        elif operator == '-<-' or operator == '-1<-':
            # n must have governor
            # get n's governor's _dependents
            # n must be equal to the last og them
            retval = lambda n, m=None, el=None: (
                _has_governor(n, positions)
                and _dependents(_governor(n, values, positions), values, positions)[-1] == n
                and predicate(_governor(n, values, positions), m, el)
            )

        # A <-N B     B is the N th-to-last child of A (the last child is <-1).
        elif operator[:3] == '->-' and operator[3:].isdigit():
            idx = int(operator[2:])
            # capture the index parameter
            retval = (
                lambda i: lambda n, m=None, el=None: (
                    _has_dependent(n, values, positions)
                    and 0
                    <= (i + len(_dependents(n, values, positions)))
                    < len(_dependents(n, values, positions))
                    and predicate(n[i + len(_dependents(n, values, positions))], m, el)
                )
            )(idx)
        # A >-N B     A is the N th-to-last child of B (the last child is >-1).
        elif operator.endswith('-<-') and operator[0].isdigit():
            idx = -int(operator[0])
            # capture the index parameter
            retval = (
                lambda i: lambda n, m=None, el=None: (
                    _has_governor(n, positions)
                    and 0 <= (i + len(_governors(n, values, positions))) < len(_governors(n, values, positions))
                    and (n is _governors(n, values, positions)[i + len(_governors(n, values, positions))])
                    and predicate(_governors(n, values, positions), m, el)
                )
            )(idx)

        # A + B       A immediately precedes B.
        elif operator == '+':
            retval = lambda n, m=None, el=None: any(
                predicate(x, m, el) for x in _immediately_after(n, values, positions)
            )
        # A - B       A immediately follows B.
        elif operator == '-':
            retval = lambda n, m=None, el=None: any(
                predicate(x, m, el) for x in _immediately_before(n, values, positions)
            )

        # A .. B      A precedes B.
        elif operator == '<|':
            retval = lambda n, m=None, el=None: any(predicate(x, m, el) for x in _after(n, values, positions))
        # A ,, B      A follows B.
        elif operator == '|>':
            retval = lambda n, m=None, el=None: any(predicate(x, m, el) for x in _before(n, values, positions))
        # todo: precedes/follows by maximum distance....

        # A .N B      A is N places to the left of B
        elif all(i == '+' for i in operator) or (
            all(i == '+' for i in operator[:-1]) and operator[-1].isdigit()
        ):
            if operator[-1].isdigit():
                num = int(operator[-1])
            else:
                num = len(operator)
            retval = lambda n, m=None, el=None: any(
                predicate(x, m, el) for x in _after(n, values, places=num)
            )

        # A ,N B      A is N places to the right of B
        elif all(i == '-' for i in operator) or (
            all(i == '-' for i in operator[:-1]) and operator[-1].isdigit()
        ):
            if operator[-1].isdigit():
                num = int(operator[-1])
            else:
                num = len(operator)
            retval = lambda n, m=None, el=None: any(
                predicate(x, m, el) for x in _before(n, values, places=num)
            )

        # A $ B       A is a sister of B (and A != B).
        elif operator == '$' or operator == '%':
            retval = lambda n, m=None, el=None: (
                _has_governor(n) and any(predicate(x, m, el) for x in _sisters(n, values, positions))
            )

        elif operator == '<$' or operator == '%..':
            raise NotImplementedError()

        # A $,, B     A is a sister of and follows B.
        elif operator == '$>' or operator == '%,,':
            raise NotImplementedError()
        else:
            raise DepgrepException('cannot interpret depgrep operator "{0}"'.format(operator))
    # now return the built function
    if negated:
        return (lambda r: (lambda n, m=None, el=None: not r(n, m, el)))(retval)
    else:
        return retval


def _depgrep_conjunction_action(_s, _l, tokens, join_char='&'):
    """
    Builds a lambda function representing a predicate on a tree node
    from the conjunction of several other such lambda functions.

    This is prototypically called for expressions like
    (`depgrep_rel_conjunction`)::

        < NP & < AP < VP

    where tokens is a list of predicates representing the relations
    (`< NP`, `< AP`, and `< VP`), possibly with the character `&`
    included (as in the example here).

    This is also called for expressions like (`depgrep_node_expr2`)::

        NP < NN
        S=s < /NP/=n : s < /VP/=v : n .. v

    tokens[0] is a depgrep_expr predicate; tokens[1:] are an (optional)
    list of segmented patterns (`depgrep_expr_labeled`, processed by
    `_depgrep_segmented_pattern_action`).
    """
    # filter out the ampersand
    tokens = [x for x in tokens if x != join_char]
    # print 'relation conjunction tokens: ', tokens
    if len(tokens) == 1:
        return tokens[0]
    else:
        return (
            lambda ts: lambda n, m=None, el=None: all(
                predicate(n, m, el) for predicate in ts
            )
        )(tokens)


def _depgrep_node_label_use_action(_s, _l, tokens):
    """
    Returns the node label used to begin a tgrep_expr_labeled.  See
    `_depgrep_segmented_pattern_action`.

    Called for expressions like (`tgrep_node_label_use`)::

        =s

    when they appear as the first element of a `tgrep_expr_labeled`
    expression (see `_depgrep_segmented_pattern_action`).

    It returns the node label.
    """
    assert len(tokens) == 1
    assert tokens[0].startswith('=')
    return tokens[0][1:]


def _macro_defn_action(_s, _l, tokens):
    """
    Builds a dictionary structure which defines the given macro.
    """
    assert len(tokens) == 3
    assert tokens[0] == '@'
    return {tokens[1]: tokens[2]}


def _depgrep_exprs_action(_s, _l, tokens):
    """
    This is the top-lebel node in a depgrep2 search string; the
    predicate function it returns binds together all the state of a
    depgrep2 search string.

    Builds a lambda function representing a predicate on a tree node
    from the disjunction of several depgrep expressions.  Also handles
    macro definitions and macro name binding, and node label
    definitions and node label binding.
    """
    if len(tokens) == 1:
        return lambda n, m=None, el=None: tokens[0](n, None, {})
    # filter out all the semicolons
    tokens = [x for x in tokens if x != ';']
    # collect all macro definitions
    macro_dict = {}
    macro_defs = [tok for tok in tokens if isinstance(tok, dict)]
    for macro_def in macro_defs:
        macro_dict.update(macro_def)
    # collect all depgrep expressions
    depgrep_exprs = [tok for tok in tokens if not isinstance(tok, dict)]

    # create a new scope for the node label dictionary
    def top_level_pred(n, m=macro_dict, el=None):
        label_dict = {}
        # bind macro definitions and OR together all depgrep_exprs
        return any(predicate(n, m, label_dict) for predicate in depgrep_exprs)

    return top_level_pred


def _depgrep_node_label_pred_use_action(_s, _l, tokens):
    """
    Builds a lambda function representing a predicate on a tree node
    which describes the use of a previously bound node label.

    Called for expressions like (`tgrep_node_label_use_pred`)::

        =s

    when they appear inside a tgrep_node_expr (for example, inside a
    relation).  The predicate returns true if and only if its node
    argument is identical the the node looked up in the node label
    dictionary using the node's label.
    """
    assert len(tokens) == 1
    assert tokens[0].startswith('=')
    node_label = tokens[0][1:]

    def node_label_use_pred(n, m=None, el=None):
        # look up the bound node using its label
        if el is None or node_label not in el:
            raise DepgrepException(f'node_label={node_label} not bound in pattern')
        node = el[node_label]
        # truth means the given node is this node
        return n is node

    return node_label_use_pred


def _depgrep_bind_node_label_action(_s, _l, tokens):
    """
    Builds a lambda function representing a predicate on a tree node
    which can optionally bind a matching node into the tgrep2 string's
    label_dict.

    Called for expressions like (`tgrep_node_expr2`)::

        /NP/
        @NP=n
    """
    # tokens[0] is a tgrep_node_expr
    if len(tokens) == 1:
        return tokens[0]
    elif len(tokens) == 2:
        return ''.join(tokens)
    else:
        # if present, tokens[1] is the character '=', and tokens[2] is
        # a tgrep_node_label, a string value containing the node label
        assert len(tokens) == 3
        assert tokens[1] == '='
        node_pred = tokens[0]
        node_label = tokens[2]

        def node_label_bind_pred(n, m=None, el=None):
            if node_pred(n, m, el):
                # bind `n` into the dictionary `l`
                if el is None:
                    problem = f'cannot bind node_label {node_label}: label_dict is None'
                    raise DepgrepException(problem)
                el[node_label] = n
                return True
            else:
                return False

        return node_label_bind_pred


def _tgrep_rel_disjunction_action(_s, _l, tokens):
    """
    Builds a lambda function representing a predicate on a tree node
    from the disjunction of several other such lambda functions.
    """
    # filter out the pipe
    tokens = [x for x in tokens if x != '|']
    # print 'relation disjunction tokens: ', tokens
    if len(tokens) == 1:
        return tokens[0]
    elif len(tokens) == 2:
        return (lambda a, b: lambda n, m=None, el=None: a(n, m, el) or b(n, m, el))(
            tokens[0], tokens[1]
        )


def _depgrep_rel_disjunction_action(_s, _l, tokens):
    """
    Builds a lambda function representing a predicate on a tree node
    from the disjunction of several other such lambda functions.
    """
    # filter out the pipe
    tokens = [x for x in tokens if x != '|']
    # print 'relation disjunction tokens: ', tokens
    if len(tokens) == 1:
        return tokens[0]
    elif len(tokens) == 2:
        return (lambda a, b: lambda n, m=None, el=None: a(n, m, el) or b(n, m, el))(
            tokens[0], tokens[1]
        )


def _depgrep_segmented_pattern_action(_s, _l, tokens):
    """
    Builds a lambda function representing a segmented pattern.

    Called for expressions like (`tgrep_expr_labeled`)::

        =s .. =v < =n

    This is a segmented pattern, a tgrep2 expression which begins with
    a node label.

    The problem is that for segemented_pattern_action (': =v < =s'),
    the first element (in this case, =v) is specifically selected by
    virtue of matching a particular node in the tree; to retrieve
    the node, we need the label, not a lambda function.  For node
    labels inside a tgrep_node_expr, we need a lambda function which
    returns true if the node visited is the same as =v.

    We solve this by creating two copies of a node_label_use in the
    grammar; the label use inside a tgrep_expr_labeled has a separate
    parse action to the pred use inside a node_expr.  See
    `_tgrep_node_label_use_action` and
    `_tgrep_node_label_pred_use_action`.
    """
    # tokens[0] is a string containing the node label
    node_label = tokens[0]
    # tokens[1:] is an (optional) list of predicates which must all
    # hold of the bound node
    reln_preds = tokens[1:]

    def pattern_segment_pred(n, m=None, el=None):
        """This predicate function ignores its node argument."""
        # look up the bound node using its label
        if el is None or node_label not in el:
            problem = 'node_label={0} not bound in pattern'.format(node_label)
            raise DepgrepException(problem)
        node = el[node_label]
        # match the relation predicates against the node
        return all(pred(node, m, el) for pred in reln_preds)

    return pattern_segment_pred


def _build_depgrep_parser(values, positions, case_sensitive=False):
    """
    Builds a pyparsing-based parser object for tokenizing and
    interpreting depgrep search strings.
    """
    depgrep_op = pyparsing.Optional('!') + pyparsing.Regex(r'[$%,.<>&-\|\+][%,.<>0-9\-\':\|]*')
    # match the node type info
    node_attr = pyparsing.Optional(pyparsing.Word('siwlxpmgfeoSIWLXPMGFEO', max=1))
    # match regex nodes
    depgrep_node_regex = node_attr + pyparsing.QuotedString(
        quoteChar='/', escChar='\\', unquoteResults=False
    )
    # match string nodes
    depgrep_qstring = node_attr + pyparsing.QuotedString(
        quoteChar='"', escChar='\\', unquoteResults=False
    )

    depgrep_qstring_icase = pyparsing.Regex('i@\\"(?:[^"\\n\\r\\\\]|(?:\\\\.))*\\"')
    depgrep_node_regex_icase = pyparsing.Regex('i@\\/(?:[^/\\n\\r\\\\]|(?:\\\\.))*\\/')
    depgrep_node_literal = pyparsing.Regex('[^][ \r\t\n;:.,&|<>()$!@%\'^=]+')
    depgrep_expr = pyparsing.Forward()
    depgrep_relations = pyparsing.Forward()
    depgrep_parens = pyparsing.Literal('(') + depgrep_expr + ')'
    depgrep_nltk_tree_pos = (
        pyparsing.Literal('N(')
        + pyparsing.Optional(
            pyparsing.Word(pyparsing.nums)
            + ','
            + pyparsing.Optional(
                pyparsing.delimitedList(pyparsing.Word(pyparsing.nums), delim=',')
                + pyparsing.Optional(',')
            )
        )
        + ')'
    )
    depgrep_node_label = pyparsing.Regex('[A-Za-z0-9]+')
    depgrep_node_label_use = pyparsing.Combine('=' + depgrep_node_label)
    # see _depgrep_segmented_pattern_action
    depgrep_node_label_use_pred = depgrep_node_label_use.copy()
    macro_name = pyparsing.Regex('[^];:.,&|<>()[$!@%\'^=\r\t\n ]+')
    macro_name.setWhitespaceChars('')
    macro_use = pyparsing.Combine('@' + macro_name)
    depgrep_node_expr = (
        depgrep_node_label_use_pred
        | macro_use
        | depgrep_nltk_tree_pos
        | depgrep_qstring_icase
        | depgrep_node_regex_icase
        | depgrep_qstring
        | depgrep_node_regex
        | '*'
        | depgrep_node_literal
    )
    depgrep_node_expr2 = (
        depgrep_node_expr
        + pyparsing.Literal('=').setWhitespaceChars('')
        + depgrep_node_label.copy().setWhitespaceChars('')
    ) | depgrep_node_expr
    depgrep_node = depgrep_parens | (
        pyparsing.Optional('\'') + depgrep_node_expr2 + pyparsing.ZeroOrMore('|' + depgrep_node_expr)
    )
    depgrep_brackets = pyparsing.Optional('!') + '[' + depgrep_relations + ']'
    depgrep_relation = depgrep_brackets | (depgrep_op + depgrep_node)
    depgrep_rel_conjunction = pyparsing.Forward()
    depgrep_rel_conjunction << (
        depgrep_relation + pyparsing.ZeroOrMore(pyparsing.Optional('&') + depgrep_rel_conjunction)
    )
    depgrep_relations << depgrep_rel_conjunction + pyparsing.ZeroOrMore('|' + depgrep_relations)
    depgrep_expr << depgrep_node + pyparsing.Optional(depgrep_relations)
    depgrep_expr_labeled = depgrep_node_label_use + pyparsing.Optional(depgrep_relations)
    depgrep_expr2 = depgrep_expr + pyparsing.ZeroOrMore(':' + depgrep_expr_labeled)
    macro_defn = pyparsing.Literal('@') + pyparsing.White().suppress() + macro_name + depgrep_expr2
    depgrep_exprs = (
        pyparsing.Optional(macro_defn + pyparsing.ZeroOrMore(';' + macro_defn) + ';')
        + depgrep_expr2
        + pyparsing.ZeroOrMore(';' + (macro_defn | depgrep_expr2))
        + pyparsing.ZeroOrMore(';').suppress()
    )

    depgrep_node_label_use.setParseAction(_depgrep_node_label_use_action)
    depgrep_node_label_use_pred.setParseAction(_depgrep_node_label_pred_use_action)
    macro_use.setParseAction(_depgrep_macro_use_action)
    depgrep_node.setParseAction(
        lambda x, y, z: _depgrep_node_action(x,
                                             y,
                                             z,
                                             positions=positions,
                                             case_sensitive=case_sensitive)
    )
    depgrep_node_expr2.setParseAction(_depgrep_bind_node_label_action)
    depgrep_parens.setParseAction(_depgrep_parens_action)
    depgrep_nltk_tree_pos.setParseAction(_depgrep_nltk_tree_pos_action)
    depgrep_relation.setParseAction(lambda x, y, z: _depgrep_relation_action(x,
                                                                             y,
                                                                             z,
                                                                             values=values,
                                                                             positions=positions))
    depgrep_rel_conjunction.setParseAction(_depgrep_conjunction_action)
    depgrep_relations.setParseAction(_depgrep_rel_disjunction_action)
    macro_defn.setParseAction(_macro_defn_action)
    # the whole expression is also the conjunction of two
    # predicates: the first node predicate, and the remaining
    # relation predicates
    depgrep_expr.setParseAction(_depgrep_conjunction_action)
    depgrep_expr_labeled.setParseAction(_depgrep_segmented_pattern_action)
    depgrep_expr2.setParseAction(functools.partial(_depgrep_conjunction_action, join_char=':'))
    depgrep_exprs.setParseAction(_depgrep_exprs_action)
    return depgrep_exprs.ignore('#' + pyparsing.restOfLine)


def depgrep_compile(depgrep_string, values=False, positions=None, case_sensitive=False):
    """
    Parses (and tokenizes, if necessary) a depgrep search string into a
    lambda function.
    """
    parser = _build_depgrep_parser(values=values, positions=positions, case_sensitive=case_sensitive)
    if isinstance(depgrep_string, bytes):
        depgrep_string = depgrep_string.decode()
    return list(parser.parseString(depgrep_string, parseAll=True))[0]
