
from antlr4.tree.Tree import TerminalNodeImpl

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