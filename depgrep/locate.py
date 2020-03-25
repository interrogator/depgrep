"""
depgrep: locators

These functions are used to traverse the dependency structures that are stored
inside pandas DataFrame objects.
"""


def ancestors(node, values, positions):
    """
    Returns the list of all nodes dominating the given tree node.
    This method will not work with leaf nodes, since there is no way
    to recover the parent.
    """
    gov_id = node[positions["g"]]
    i = node[positions["i"]]
    n = node[positions["_n"]]
    out = [values[n - i + gov_id]]
    # in case there is some kind of missing link
    depth = 10
    while gov_id and depth:
        # confirm this calculation, may need adjustment
        row = values[n - i + gov_id]
        gov_id = row[positions["g"]]
        out.append(row)
        depth -= 1
    return out


def descendants(node, values, positions, depth=5):
    """
    Recursively get all dependents
    """
    out = list()
    if not depth:
        return out
    # cut down to sentence but only first time
    sent = sentence(node, values, positions) if depth == 5 else values
    # all dependents of this node
    dep_of_node = sent[sent[:, positions["g"]] == node[positions["i"]]]
    # for each, run recursively
    for dep in dep_of_node:
        out.append(dep)
        out += descendants(dep, sent, positions, depth - 1)
    return out


def before(node, values, positions, places=False):
    """
    Returns the set of all nodes that are before the given node.
    """
    n = node[positions["_n"]]
    if places is not False:
        try:
            return [values[n - places]]
        except IndexError:
            return list()
    i = node[positions["i"]]
    return values[n + 1 - i : n]


def immediately_before(node, values, positions):
    """
    Returns the set of all nodes that are immediately after the given
    node.

    Tree node A immediately follows node B if the first terminal
    symbol (word) produced by A immediately follows the last
    terminal symbol produced by B.
    """
    i = node[positions["_n"]] - 1
    try:
        return [values[i]]
    except:
        return list()


def after(node, values, positions, places=False):
    """
    Returns the set of all nodes that are after the given node.
    """

    n = node[positions["_n"]]
    if places is not False:
        try:
            return [values[n + places]]
        except IndexError:
            return list()
    sent_len = node[positions["sent_len"]]
    # i = node[positions["i"]]
    return values[n + 1 : n + sent_len - n]


def immediately_after(node, values, positions):
    """
    Returns the set of all nodes that are immediately after the given
    node.

    Tree node A immediately follows node B if the first terminal
    symbol (word) produced by A immediately follows the last
    terminal symbol produced by B.
    """
    i = node[positions["_n"]] + 1
    try:
        return [values[i]]
    except:
        return list()


def governor(row, values, positions):
    """
    Get governor of row as row from values
    """
    g = row[positions["g"]]
    if not g:
        # return row
        return "ROOT"
    else:
        i = row[positions["i"]]
        n = row[positions["_n"]]
        return values[n - i + g]


def sentence(row, values, positions):
    slen = row[positions["sent_len"]]
    rn = row[positions["_n"]]
    i = row[positions["i"]]
    start = rn - i + 1
    end = start + slen
    return values[start:end]


def dependents(row, values, positions):
    """
    Get list of dependents of row as rows from values
    """
    sent = sentence(row, values, positions)
    return sent[sent[:, positions["g"]] == row[positions["i"]]]


def governors(row, values, positions):
    """
    Sometimes we need a list of _governors, but only one is possible
    So, here we just return a one-item list
    """
    return [governor(row, values, positions)]


def sisters(node, values, positions):
    """
    Give a list of tokens sharing a governor, but not node
    """
    sent = sentence(node, values, positions)
    i = node[positions["i"]]
    same_gov = sent[sent[:, positions["g"]] == node[positions["g"]]]
    return [r for r in same_gov if r[positions["i"]] != i]
