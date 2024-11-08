def reformat_tree(jplace):
    """ Convert edge-numbered Newick to (more) conventional Newick.
    We expect an input tree of the form:
    ((A:.01{0}, B:.01{1})1.0:.01{3}, C:.01{4}){5};
    (in jplace['tree']- jplace should be the output of load_jplace)
    and wish to obtain a tree with unique internal node identifiers,
    but without edge numbering. We also don't care about internal node
    support values.
    
    We achieve this by reidentifying each node (internal and leaf) 
    with its edge number (that is, the edge number of the edge 
    connecting it to its parent:)
    (("0":.01, "1":.01)"3":.01, "4":.01)"5"; 
    The resulting Newick string is read into an ete3.Tree and that is 
    returned as the first value.
    We also keep track of the original names (for leaves) or 
    support values (for internal nodes) and return two dictionaries:
    one mapping original leaf names to new ids, and one mapping
    new ids to collections of the original names of all descendant
    leaves. Note the edge numbers appear in these dictionaries as strings,
    whether they are keys or values, eg {'0': A, ...}. 
    """
    tree_string = jplace['tree']
    new_id_to_old_value = {}
    for match in edge_number_pattern.finditer(tree_string):
        value, distance, edge_number = match.groups()
        new_id_to_old_value[edge_number] = value

    relabel = lambda match: match.group(3) + ':' + match.group(2)

    new_tree, _ = edge_number_pattern.subn(relabel,tree_string)
    new_tree = terminal_edge_number_pattern.sub(
        lambda match: '%s;' % match.group(1),new_tree
    )

    new_tree = Tree(new_tree, format=1)
    
    leaf_to_new_id = {}
    new_id_to_old_leaf_names = {} 

    for node in new_tree.traverse(strategy='levelorder'):
        name = node.name
        if node.is_leaf():
            leaf_to_new_id[new_id_to_old_value[name]] = name
        old_names_of_leaves = []
        for leaf in node.get_leaves():
            old_names_of_leaves.append(
                new_id_to_old_value[leaf.name]
            )
        new_id_to_old_leaf_names[name] = old_names_of_leaves
        
    return new_tree, leaf_to_new_id, new_id_to_old_leaf_names


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-jp", "--jplace", help="jplace alignment file", type=str)
    args = parser.parse_args()
    new_tree, leaf_to_new_id, new_id_to_old_leaf_names = reformat_tree(args.jplace)
