# Sisyphean skill — evaluate a math expression safely using math.* functions
import sys
import math


SAFE_NS = {name: getattr(math, name) for name in dir(math) if not name.startswith("_")}
SAFE_NS["abs"] = abs
SAFE_NS["round"] = round
SAFE_NS["min"] = min
SAFE_NS["max"] = max
SAFE_NS["sum"] = sum
SAFE_NS["pow"] = pow
# Add uppercase aliases so SQRT(144) works the same as sqrt(144)
SAFE_NS.update({k.upper(): v for k, v in list(SAFE_NS.items())})


def evaluate(expr: str) -> str:
    try:
        result = eval(expr, {"__builtins__": {}}, SAFE_NS)
        # Return int if result is a whole float
        if isinstance(result, float) and result.is_integer():
            return str(int(result))
        return str(result)
    except ZeroDivisionError:
        return "Error: division by zero"
    except (SyntaxError, NameError, TypeError, ValueError) as e:
        return f"Error: {e}"
    except Exception as e:
        return f"Error: {e}"


def main():
    if len(sys.argv) < 2:
        print("Usage: python skills/calc.py EXPRESSION")
        print("       python skills/calc.py sqrt 144")
        return

    # Join all args — supports both quoted and space-separated forms
    raw = " ".join(sys.argv[1:])

    # Try as-is first; if that fails and we have exactly two tokens (func arg),
    # retry as func(arg)  e.g. "sqrt 144" -> "sqrt(144)"
    result = evaluate(raw)
    if result.startswith("Error:") and len(sys.argv) == 3:
        alt = f"{sys.argv[1]}({sys.argv[2]})"
        alt_result = evaluate(alt)
        if not alt_result.startswith("Error:"):
            result = alt_result
    print(result)


if __name__ == "__main__":
    main()
