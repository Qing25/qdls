
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

def get_sparql_nodes(tree, parser):
    """遍历语法树 返回树字符串、节点列表、  

    Args:
        tree: 
        parser: 
    """
    parts, parents = [], []
    parse_layer(tree, parser.ruleNames, parts, parents)
    s = tree.toStringTree(recog=parser)
    return s, parts, parents 

def split_sparql(query, lexer=None):
    """使用lexer将查询语句分词（语义单元级别

    Args:
        query: 查询字符串，可以包含语法错误
        lexer: pygments的lexer. Defaults to cy_lexer.

    Returns:
        a list of strings 
    """
    if lexer is None:
        try:
            from pygments.lexers.rdf import SparqlLexer as SPLexer
            lexer = SPLexer()
        except:
            raise Exception(f"lexer is not set!")

    pred_units = []
    for tag, substr in lexer.get_tokens(query):
        if tag is Token.Error:
            # return -1
            pred_units.append("错")
        # print(tag, substr)
        if substr.strip() != "":
            pred_units.append(substr)

    return pred_units

