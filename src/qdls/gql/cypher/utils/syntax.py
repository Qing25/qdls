
from ..grammar.CypherLexer import CypherLexer
from ..grammar.CypherListener import CypherListener
from ..grammar.CypherParser import CypherParser

from antlr4 import InputStream,CommonTokenStream
from antlr4.error.ErrorListener import ErrorListener


class MyParserErrorListener(ErrorListener):
    def syntaxError(self, recognizer, offendingSymbol, line, column, msg, e):
        raise Exception(f"Syntax Error:{msg}\n{e}\n{column}\n{line}\n{offendingSymbol}")

class MyLexerErrorListener(ErrorListener):
    def syntaxError(self, recognizer, offendingSymbol, line, column, msg, e):
        raise Exception(f"Lexer Error:{msg}\n{e}\n{column}\n{line}\n{offendingSymbol}")

def syntax_check(cypher):
    """
        using antlr4 to pre-check if generated cypher is syntax correct
        using self-defined ErrorListener to avoid console output
    """
    input_stream = InputStream(cypher)
    lexer = CypherLexer(input_stream)
    lexer._listeners = []
    lexer.addErrorListener(MyLexerErrorListener())
    stream = CommonTokenStream(lexer)
    parser = CypherParser(stream)
    parser._listeners = []
    parser.addErrorListener(MyParserErrorListener())
    try:
        tree = parser.oC_Cypher()
    except Exception as e:
        # print(e)
        return False, e, parser
    return True, tree, parser


if __name__ == '__main__':
    
    s1 = "match (q:SS return q limit 1"

    flag, _ = syntax_check(s1)

    print(flag)
    print("error msg:", _)

    """
    Syntax Error:no viable alternative at input 'match (q:SS return'
    None
    12
    1
    [@7,12:17='return',<62>,1:12]

    根据offendingSymbol可以定位到错误
    """
