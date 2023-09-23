
from ..grammar.SparqlLexer import SparqlLexer
from ..grammar.SparqlListener import SparqlListener
from ..grammar.SparqlParser import SparqlParser

from antlr4 import InputStream,CommonTokenStream
from antlr4.error.ErrorListener import ErrorListener


class MyParserErrorListener(ErrorListener):
    def syntaxError(self, recognizer, offendingSymbol, line, column, msg, e):
        raise Exception(f"Syntax Error:{msg}\n{e}\n{column}\n{line}\n{offendingSymbol}")

class MyLexerErrorListener(ErrorListener):
    def syntaxError(self, recognizer, offendingSymbol, line, column, msg, e):
        raise Exception(f"Lexer Error:{msg}\n{e}\n{column}\n{line}\n{offendingSymbol}")

def syntax_check(sparql):
    """
        using antlr4 to pre-check if generated sparql is syntax correct
        using self-defined ErrorListener to avoid console output
    """
    input_stream = InputStream(sparql)
    lexer = SparqlLexer(input_stream)
    lexer._listeners = []
    lexer.addErrorListener(MyLexerErrorListener())
    stream = CommonTokenStream(lexer)
    parser = SparqlParser(stream)
    parser._listeners = []
    parser.addErrorListener(MyParserErrorListener())
    try:
        tree = parser.query()
    except Exception as e:
        # print(e)
        return False, e, parser
    return True, tree, parser


if __name__ == '__main__':
    
    s1 = "select {?e ?p 'asd'}"

    flag, _ = syntax_check(s1)

    print(flag)
    print("error msg:", _)

