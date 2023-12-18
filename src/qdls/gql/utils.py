
from antlr4.tree.Tree import TerminalNodeImpl
from nltk import Tree


def parse_layer(tree, rule_names, parts, parents, indent = 0):
    """ 遍历树，保存叶子结点 """
    if tree.getText() == "<EOF>":
        return
    elif isinstance(tree, TerminalNodeImpl):
        # print("{0}TOKEN='{1}'".format("  " * indent, tree.getText()))
        # layer[indent].append(tree.getText())
        parts.append(tree.getText())
        parents.append(tree.getParent())
    elif tree.children is None:
        # print("none children", tree)
        return
    else:
        # print("{0}{1}".format("  " * indent, rule_names[tree.getRuleIndex()]))
        for child in tree.children:
            parse_layer(child, rule_names, parts, parents, indent + 1)



def save_ast(tree, parser, path):
    """ 
        将 antlr tree 保存为 png or pdf 
        ```
        tree, parser = parse_cypher(hop1['cypher'], True)
        save_ast(tree, parser, "hop1_cypher.pdf")
        ```
    """
    try:
        import svgling, cairosvg
    except Exception as e:
        print("Try ` pip install svgling cairosvg `")
        raise e 

    tree_string = tree.toStringTree(recog=parser)
    img = svgling.draw_tree(Tree.fromstring(tree_string))
    svg = img.get_svg()
    if path.endswith("pdf"):
        cairosvg.svg2pdf(svg.tostring(),  write_to=path, unsafe=False)
    elif path.endswith("png"):
        cairosvg.svg2png(svg.tostring(),  write_to=path)
    else:
        raise Exception(path, "is not pdf or png")