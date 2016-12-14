import re
grammar={
        "W" : ["%C%%T%","%C%%T%","%C%%X%","%C%%D%%F%","%C%%V%%F%%T%","%C%%D%%F%%U%","%C%%T%%U%","%I%%T%","%I%%C%%T%","%A%"],
        "A" : ["%K%%V%%K%%V%tion"],
        "K" : ["b","c","d","f","g","j","l","m","n","p","qu","r","s","t","v","s%P%"],
        "I" : ["ex","in","un","re","de"],
        "T" : ["%V%%F%","%V%%E%e"],
        "U" : ["er","ish","ly","en","ing","ness","ment","able","ive"],
        "C" : ["b","c","ch","d","f","g","h","j","k","l","m","n","p","qu","r","s","sh","t","th","v","w","y","s%P%","%R%r","%L%l"],
        "E" : ["b","c","ch","d","f","g","dg","l","m","n","p","r","s","t","th","v","z"],
        "F" : ["b","tch","d","ff","g","gh","ck","ll","m","n","n","ng","p","r","ss","sh","t","tt","th","x","y","zz","r%R%","s%P%","l%L%"],
        "P" : ["p","t","k","c"],
        "Q" : ["b","d","g"],
        "L" : ["b","f","k","p","s"],
        "R" : ["%P%","%Q%","f","th","sh"],
        "V" : ["a","e","i","o","u"],
        "D" : ["aw","ei","ow","ou","ie","ea","ai","oy"],
        "X" : ["e","i","o","aw","ow","oy"]
    };

def splitted(word):
    return re.findall('%[A-Z]%|[a-z]+',word)
def get_nonterminal(rule):
    mt = re.match('%([A-Z])%',rule)
    if mt:
        return mt.group(1)
    else:
        return None

def get_terminal(rule):
    mt = re.match('[a-z]+',rule)
    if mt:
        return mt.group(0)
    else:
        return None

def accept(grammar,word,string):
    '''grammar: as in example
    word: grammar pattern to match with
    string: string to accept against word pattern'''
    rules = splitted(word)
    if(len(rules)==1):
        rule = rules[0]
        terminal = get_terminal(rule)
        if(terminal):
            if(string.startswith(terminal)):
                return [terminal]
            else:
            	return []
        nonterminal = get_nonterminal(rule)
        accepted = []
        for rule in grammar[nonterminal]:
            accepted += accept(grammar,rule,string)
        return accepted
    else:
        accepted = []
        rule = rules[0]
        rest_rules = ''.join(rules[1:])
        for prefix in accept(grammar,rule,string):
            postfixes = accept(grammar,rest_rules,string[len(prefix):])
            accepted += list(map(lambda s:prefix+s,postfixes))
        return accepted

def wakaba_accept(string):
	return string in accept(grammar,"%W%",string)