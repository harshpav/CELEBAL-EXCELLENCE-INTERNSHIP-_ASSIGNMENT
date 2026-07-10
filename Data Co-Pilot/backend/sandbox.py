"""
sandbox.py - Safe Python code execution for data analysis.
"""
import io
import base64
import traceback
import contextlib
import threading
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

EXEC_TIMEOUT = 30  # seconds

BLOCKED = [
    "import os", "import sys", "import subprocess", "import socket",
    "os.system", "os.popen", "subprocess", "__import__('os')",
    "open(", "eval(", "compile(", "requests", "urllib",
]

def is_safe(code: str) -> tuple:
    low = code.lower()
    for b in BLOCKED:
        if b.lower() in low:
            return False, f"Blocked: '{b}'"
    return True, ""

def execute_code(code: str, df: pd.DataFrame) -> dict:
    safe, reason = is_safe(code)
    if not safe:
        return {"success": False, "output": "", "chart": None, "error": reason}

    stdout_buf = io.StringIO()
    result_container = {}

    ns = {
        "pd": pd, "np": np, "plt": plt, "df": df.copy(),
        "__builtins__": {
            "print": print, "len": len, "range": range,
            "enumerate": enumerate, "zip": zip, "list": list,
            "dict": dict, "set": set, "tuple": tuple,
            "str": str, "int": int, "float": float, "bool": bool,
            "round": round, "abs": abs, "min": min, "max": max,
            "sum": sum, "sorted": sorted, "isinstance": isinstance,
            "type": type, "__import__": __import__,
        },
    }

    try:
        import seaborn as sns
        ns["sns"] = sns
    except ImportError:
        pass

    def _run():
        try:
            plt.close("all")
            with contextlib.redirect_stdout(stdout_buf):
                exec(code, ns)  # noqa: S102

            fig = plt.gcf()
            chart_b64 = None
            if fig.get_axes():
                buf = io.BytesIO()
                fig.savefig(buf, format="png", bbox_inches="tight", dpi=120)
                buf.seek(0)
                chart_b64 = base64.b64encode(buf.read()).decode()
                plt.close("all")

            result_container["result"] = {
                "success": True,
                "output": stdout_buf.getvalue(),
                "chart": chart_b64,
                "error": None,
            }
        except Exception as e:
            plt.close("all")
            result_container["result"] = {
                "success": False,
                "output": stdout_buf.getvalue(),
                "chart": None,
                "error": f"{type(e).__name__}: {e}\n{traceback.format_exc()}",
            }

    thread = threading.Thread(target=_run, daemon=True)
    thread.start()
    thread.join(timeout=EXEC_TIMEOUT)

    if thread.is_alive():
        plt.close("all")
        return {
            "success": False,
            "output": "",
            "chart": None,
            "error": f"Execution timed out after {EXEC_TIMEOUT} seconds.",
        }

    return result_container.get("result", {
        "success": False,
        "output": "",
        "chart": None,
        "error": "Unknown execution error.",
    })
