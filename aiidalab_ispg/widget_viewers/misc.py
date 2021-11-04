"""Some useful classes used acrross the repository."""
import io
import tokenize

import ipywidgets as ipw
from traitlets import Unicode


class CopyToClipboardButton(ipw.Button):
    """Button to copy text to clipboard."""

    value = Unicode(allow_none=True)  # Traitlet that contains a string to copy.

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        super().on_click(self.copy_to_clipboard)

    def copy_to_clipboard(self, change=None):  # pylint:disable=unused-argument
        """Copy text to clipboard."""
        from IPython.display import Javascript, display

        javas = Javascript(
            """
           function copyStringToClipboard (str) {{
               // Create new element
               var el = document.createElement('textarea');
               // Set value (string to be copied)
               el.value = str;
               // Set non-editable to avoid focus and move outside of view
               el.setAttribute('readonly', '');
               el.style = {{position: 'absolute', left: '-9999px'}};
               document.body.appendChild(el);
               // Select text inside element
               el.select();
               // Copy text to clipboard
               document.execCommand('copy');
               // Remove temporary element
               document.body.removeChild(el);
            }}
            copyStringToClipboard("{selection}");
       """.format(
                selection=self.value
            )
        )  # For the moment works for Chrome, but doesn't work for Firefox.
        if self.value:  # If no value provided - do nothing.
            display(javas)


class ReversePolishNotation:
    """Class defining operations for RPN conversion"""

    # adapted from:
    # Author: Alaa Awad
    # Description: program converts infix to postfix notation
    # https://gist.github.com/awadalaa/7ef7dc7e41edb501d44d1ba41cbf0dc6
    def __init__(self, operators, additional_operands=None):
        self.operators = operators
        self.additional_operands = additional_operands

    def haslessorequalpriority(self, opa, opb):
        """Priority of the different operators"""
        if opa not in self.operators:
            return False
        if opb not in self.operators:
            return False
        return self.operators[opa]["priority"] <= self.operators[opb]["priority"]

    def is_operator(self, opx):
        """Identifies operators"""
        return opx in self.operators

    @staticmethod
    def isopenparenthesis(operator):
        """Identifies open paretheses."""
        return operator == "("

    @staticmethod
    def iscloseparenthesis(operator):
        """Identifies closed paretheses."""
        return operator == ")"

    def convert(self, expr):
        """Convert expression to postfix."""
        stack = []
        output = []
        for char in expr:
            if self.is_operator(char) or char in ["(", ")"]:
                if self.isopenparenthesis(char):
                    stack.append(char)
                elif self.iscloseparenthesis(char):
                    operator = stack.pop()
                    while not self.isopenparenthesis(operator):
                        output.append(operator)
                        operator = stack.pop()
                else:
                    while stack and self.haslessorequalpriority(char, stack[-1]):
                        output.append(stack.pop())
                    stack.append(char)
            else:
                output.append(char)
        while stack:
            output.append(stack.pop())
        return output

    @staticmethod
    def parse_infix_notation(condition):
        """Convert a string containing the expression into a list of operators and operands."""
        condition = [
            token[1]
            for token in tokenize.generate_tokens(
                io.StringIO(condition.strip()).readline
            )
            if token[1]
        ]

        result = []
        open_bracket = False

        # Merging lists.
        for element in condition:
            if element == "[":
                res = "["
                open_bracket = True
            elif element == "]":
                res += "]"
                result.append(res)
                open_bracket = False
            elif open_bracket:
                res += element
            else:
                result.append(element)
        return result

    def execute(self, expression):
        """Execute the provided expression."""

        def is_number(string):
            """Check if string is a number. """
            try:
                float(string)
                return True
            except ValueError:
                return False

        stack = []
        stackposition = -1
        infix_expression = self.parse_infix_notation(expression)
        for ope in self.convert(infix_expression):
            # Operands.
            if is_number(ope):
                stack.append(float(ope))
                stackposition += 1
            elif ope in self.operators:
                nargs = self.operators[ope]["nargs"]
                arguments = [
                    stack[stackposition + indx] for indx in list(range(-nargs + 1, 1))
                ]
                stack[stackposition] = self.operators[ope]["function"](*arguments)
                del stack[stackposition - nargs + 1 : stackposition]
                stackposition -= nargs - 1
            else:
                if self.additional_operands and ope in self.additional_operands:
                    stack.append(self.additional_operands[ope])
                else:
                    stack.append(ope)
                stackposition += 1
        return stack[0] if stack else []
