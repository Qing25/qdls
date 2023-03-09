
from pygments.lexers import get_lexer_by_name
from pygments.lexers.graph import CypherLexer
from pygments.token import *
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

def get_cypher_nodes(tree, parser):
    """遍历语法树 返回树字符串、节点列表、  

    Args:
        tree: 
        parser: 

    """
    parts, parents = [], []
    parse_layer(tree, parser.ruleNames, parts, parents)
    s = tree.toStringTree(recog=parser)
    return s, parts, parents 

def split_query(query, lexer=None):
    """使用lexer将查询语句分词（语义单元级别

    Args:
        query: 查询字符串，可以包含语法错误
        lexer: pygments的lexer. Defaults to cy_lexer.

    Returns:
        a list of strings 
    """
    if lexer is None:
        try:
            lexer = get_lexer_by_name('py2neo.cypher')
        except:
            raise Exception(f"lexer is not set! Using py2neo failed, run `pip install py2neo`")

    pred_units = []
    for tag, substr in lexer.get_tokens(query):
        if tag is Token.Error:
            # return -1
            pred_units.append("错")
        # print(tag, substr)
        if substr.strip() != "":
            pred_units.append(substr)

    return pred_units