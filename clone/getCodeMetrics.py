import javalang,sys
from utils import get_sequence as func
from collections import Counter

def getMetricsVec(code):
    tokens = javalang.tokenizer.tokenize(code)
    parser = javalang.parser.Parser(tokens)
    tree = parser.parse_member_declaration()

    def trans_to_sequences(ast):
        sequence = []
        func(ast, sequence)
        return sequence

    corpus = trans_to_sequences(tree)
    c = Counter(corpus)
    numVariablesDeclared = c["VariableDeclarator"]
    numOperands = c["BinaryOperation"]*2+c["TernaryExpression"]*3+c["++"]+c["--"]+c["!"]+c["~"]
    numArgs = c["FormalParameter"]
    numExpressions = c["StatementExpression"]
    numOperators = c["BinaryOperation"]+c["TernaryExpression"]
    numLoops = c["ForStatement"]+c["WhileStatement"]
    numExceptThrown = c["TryStatement"]
    numExceptRefer = c["CatchClause"]
    cyclomaticNumber = c["IfStatement"]+c["WhileStatement"]+c["ForStatement"]+c["TryStatement"]+c["SwitchStatementCase"]
    return [numVariablesDeclared,numOperators,numArgs,numExpressions,numOperands,numLoops,numExceptThrown,numExceptRefer,cyclomaticNumber]
