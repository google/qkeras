# Copyright 2020 Google LLC
#
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""Parses PLA format usig ply."""
from ply import yacc
from ply import lex
import numpy as np

_1 = 1
_0 = 2
_X = 3
_U = 0

NOT = {_0: _1, _1: _0, _X: _U, _U: _U}

class PLA:
  def __init__(self):
    self.pla_i = []
    self.pla_o = []

pla = PLA()

tokens = [
  "I",
  "O",
  "MV",
  "ILB",
  "OB",
  "P",
  "L",
  "E",
  "TYPE",
  "SYMBOL",
  "NUMBER",
  "NEWLINE"
]

t_ignore = " \t|"
t_I = r"\.[iI]"
t_O = r"\.[oO]"
t_MV = r"\.[mM][vV]"
t_ILB = r"\.[iI][lL][bB]"
t_OB = r"\.[oO][bB]"
t_P = r"\.[pP]"
t_L = r"\.[lL]"
t_E = r"\.[eE]"
t_TYPE = r"\.type"
t_SYMBOL = r"[a-zA-Z_][a-zA-Z0-9_\<\>\-\$]*"

def t_NUMBER(t):
  r"[\d\-]+"
  return t

def t_NEWLINE(t):
  r"\n+"
  t.lexer.lineno += t.value.count("\n")
  return t

def t_error(t):
  print("Illegal character '{}'".format(t.value))
  t.lexer.skip(1)

lex.lex()

def p_pla(p):
  """pla : pla_declarations pla_table pla_end"""

def p_pla_declarations(p):
  """pla_declarations : pla_declarations pla_declaration
                      | pla_declaration"""

def p_pla_declaration(p):
  """pla_declaration : I NUMBER NEWLINE
                     | O NUMBER NEWLINE
                     | P NUMBER NEWLINE
                     | MV number_list NEWLINE
                     | ILB symbol_list NEWLINE
                     | OB symbol_list NEWLINE
                     | L NUMBER symbol_list NEWLINE
                     | TYPE SYMBOL NEWLINE
  """
  token = p[1].lower()
  if token == ".i":
    pla.ni = int(p[2])
  elif token == ".o":
    pla.no = int(p[2])
  elif token == ".mv":
    pla.mv = [int(v) for v in p[2]]
  elif token == ".ilb":
    pla.ilb = p[2]
  elif token == ".ob":
    pla.ob = p[2]
  elif token == ".l":
    pla.label = p[2]
  elif token == ".type":
    pla.set_type = p[2]


def p_pla_table(p):
  """pla_table : pla_table number_symbol_list NEWLINE
               | number_symbol_list NEWLINE"""
  if len(p[1:]) == 3:
    line = "".join(p[2])
  else:
    line = "".join(p[1])

  assert hasattr(pla, "ni") and hasattr(pla, "no")

  # right now we only process binary functions

  line = [_1 if v == "1" else _0 if v == "0" else _X for v in line]

  pla.pla_i.append(line[0:pla.ni])
  pla.pla_o.append(line[pla.ni:])


def p_pla_end(p):
  """pla_end : E opt_new_line"""
  pass


def p_opt_new_line(p):
  """opt_new_line : NEWLINE
                  |
  """
  pass


def p_number_list(p):
  """number_list : number_list NUMBER
                 | NUMBER
  """
  if len(p[1:]) == 2:
    p[0] = p[1] + [p[2]]
  else:
    p[0] = [p[1]]


def p_symbol_list(p):
  """symbol_list : symbol_list SYMBOL
                 | SYMBOL
  """
  if len(p[1:]) == 2:
    p[0] = p[1] + [p[2]]
  else:
    p[0] = [p[1]]


def p_number_symbol_list(p):
  """number_symbol_list : number_symbol_list number_or_symbol
                        | number_or_symbol
  """
  if len(p[1:]) == 2:
    p[0] = p[1] + [p[2]]
  else:
    p[0] = [p[1]]


def p_number_or_symbol(p):
  """number_or_symbol : NUMBER
                      | SYMBOL
  """
  p[0] = p[1]


def p_error(p):
  print("Error text at {}".format(p)) #p.value))

yacc.yacc()

def get_tokens(fn):
  lex.input("".join(open(fn).readlines()))
  return lex.token

def parse(fn):
  yacc.parse("".join(open(fn).readlines()))

  pla.pla_i = np.array(pla.pla_i)
  pla.pla_o = np.array(pla.pla_o)

  return pla
