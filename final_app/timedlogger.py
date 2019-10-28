import time
import datetime
import inspect
import logging

def fstr(fstring_text, locals, globals=None):
    """
    Dynamically evaluate the provided fstring_text

    Sample usage:
        format_str = "{i}*{i}={i*i}"
        i = 2
        fstr(format_str, locals()) # "2*2=4"
        i = 4
        fstr(format_str, locals()) # "4*4=16"
        fstr(format_str, {"i": 12}) # "10*10=100"
    """
    locals = locals or {}
    globals = globals or {}
    ret_val = eval(f'f"{fstring_text}"', locals, globals)
    return ret_val

class timelogged(object):
    """
    Decorator class for logging function start, completion, and elapsed time.
    """

    def __init__(
        self,
        desc_text="'{desc_detail}' call to {fn.__name__}()",
        desc_detail="",
        start_msg="Выполняется {desc_text}...",
        success_msg="Готово: {desc_text}  {elapsed}",
        log_fn=logging.info,
        **addl_kwargs,
    ):
        """ All arguments optional """
        self.context = addl_kwargs.copy()  # start with addl. args
        self.context.update(locals())  # merge all constructor args
        self.context["elapsed"] = None
        self.context["start"] = time.time()

    def re_eval(self, context_key: str):
        """ Evaluate the f-string in self.context[context_key], store back the result """
        self.context[context_key] = fstr(self.context[context_key], locals=self.context)

    def elapsed_str(self):
        """ Return a formatted string, e.g. '(HH:MM:SS elapsed)' """
        seconds = time.time() - self.context["start"]
        delta = datetime.timedelta(seconds=seconds)
        str_delta = ""

        seconds = delta.total_seconds()
        hours = int(seconds // 3600)
        minutes = int((seconds % 3600) // 60)
        seconds = int(seconds % 60)

        if hours:
            str_delta += str(minutes) + "h "

        if minutes:
            str_delta += str(minutes) + "m "

        if seconds:
            str_delta += str(seconds) + "s "

        milliseconds = round(delta.microseconds / 1000)
        str_delta += str(milliseconds) + "ms"
        
        return "(за {})".format(str_delta)

    def __call__(self, fn):
        """ Call the decorated function """

        def wrapped_fn(*args, **kwargs):
            """
            The decorated function definition. Note that the log needs access to 
            all passed arguments to the decorator, as well as all of the function's
            native args in a dictionary, even if args are not provided by keyword.
            If start_msg is None or success_msg is None, those log entries are skipped.
            """
            self.context["fn"] = fn
            fn_arg_names = inspect.getfullargspec(fn).args
            for x, arg_value in enumerate(args, 0):
                self.context[fn_arg_names[x]] = arg_value
            self.context.update(kwargs)
            desc_detail_fn = None
            log_fn = self.context["log_fn"]
            # If desc_detail is callable, evaluate dynamically (both before and after)
            if callable(self.context["desc_detail"]):
                desc_detail_fn = self.context["desc_detail"]
                self.context["desc_detail"] = desc_detail_fn()

            # Re-evaluate any decorator args which are fstrings
            self.re_eval("desc_detail")
            self.re_eval("desc_text")
            # Remove 'desc_detail' if blank or unused
            self.context["desc_text"] = self.context["desc_text"].replace("'' ", "")
            self.re_eval("start_msg")
            if self.context["start_msg"]:
                # log the start of execution
                log_fn(self.context["start_msg"])
            ret_val = fn(*args, **kwargs)
            if desc_detail_fn:
                # If desc_detail callable, then reevaluate
                self.context["desc_detail"] = desc_detail_fn()
            self.context["elapsed"] = self.elapsed_str()
            # log the end of execution
            log_fn(fstr(self.context["success_msg"], locals=self.context))
            return ret_val

        return wrapped_fn
