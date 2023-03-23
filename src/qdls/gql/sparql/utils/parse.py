
from pygments.token import *
from antlr4.tree.Tree import TerminalNodeImpl
from antlr4 import InputStream, CommonTokenStream

from ..grammar.SparqlParser import SparqlParser
from ..grammar.SparqlLexer import SparqlLexer

from ...utils import parse_layer

def parse_sparql(sparql, tree_only=False):
    """ 返回输入sparql语句的AST树 
        AST 树、序列化的树
        便利后得到的 每个token 及其对应的父节点
    """
    input_stream = InputStream(sparql)
    # print(input_stream)
    lexer = SparqlLexer(input_stream)
    stream = CommonTokenStream(lexer)
    parser = SparqlParser(stream)
    tree = parser.query()
    tree.toStringTree()
    if tree_only:
        return tree, parser  
        
    parts, parents = [], []
    # print(tree)
    # traverse(tree, parser.ruleNames)
    parse_layer(tree, parser.ruleNames, parts, parents)
    print("after parsing", tree)
    for token, par in zip(parts, parents):
        print(token, "\t", par)
    s = tree.toStringTree(recog=parser)
    return tree, s, parts, parents 