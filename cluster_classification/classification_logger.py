from enum import Enum
from typing import override, Dict, Any, Mapping
import logging
import os


# Yoinked from stackoverflow.com/questions/2183233/how-to-add-a-custom-loglevel-to-pythons-logging-facility/35804945#35804945
def addLoggingLevel(levelName, levelNum, methodName=None):
    """Comprehensively adds a new logging level to the `logging` module and the
    currently configured logging class.

    `levelName` becomes an attribute of the `logging` module with the value
    `levelNum`. `methodName` becomes a convenience method for both `logging`
    itself and the class returned by `logging.getLoggerClass()` (usually just
    `logging.Logger`). If `methodName` is not specified, `levelName.lower()` is
    used.

    To avoid accidental clobberings of existing attributes, this method will
    raise an `AttributeError` if the level name is already an attribute of the
    `logging` module or if the method name is already present 

    Example
    -------
    >>> addLoggingLevel('TRACE', logging.DEBUG - 5)
    >>> logging.getLogger(__name__).setLevel("TRACE")
    >>> logging.getLogger(__name__).trace('that worked')
    >>> logging.trace('so did this')
    >>> logging.TRACE
    5

    """
    if not methodName:
        methodName = levelName.lower()

    if hasattr(logging, levelName):
       raise AttributeError('{} already defined in logging module'.format(levelName))
    if hasattr(logging, methodName):
       raise AttributeError('{} already defined in logging module'.format(methodName))
    if hasattr(logging.getLoggerClass(), methodName):
       raise AttributeError('{} already defined in logger class'.format(methodName))

    # This method was inspired by the answers to Stack Overflow post
    # http://stackoverflow.com/q/2183233/2988730, especially
    # http://stackoverflow.com/a/13638084/2988730
    def logForLevel(self, message, *args, **kwargs):
        if self.isEnabledFor(levelNum):
            self._log(levelNum, message, args, **kwargs)
    def logToRoot(message, *args, **kwargs):
        logging.log(levelNum, message, *args, **kwargs)

    logging.addLevelName(levelNum, levelName)
    setattr(logging, levelName, levelNum)
    setattr(logging.getLoggerClass(), methodName, logForLevel)
    setattr(logging, methodName, logToRoot)

addLoggingLevel('TRACE', logging.DEBUG - 5)
addLoggingLevel('CRITICAL', logging.FATAL - 5)

class RecordShape(Enum):
    """Represents the "shape" of the record should use"""
    OPEN = 0
    """'Opens' a record, which will be matched with a closing record"""
    CLOSE = 1
    """'Closes' an 'opened' record."""
    SINGLE = 2
    """Standalone Record"""    

class XMLFormatter(logging.Formatter):
    """Formats the log record for printing to the log file.
    
    Overrides the default format method.
    """

    def __init__(self):
        super(XMLFormatter, self).__init__()

    @override
    def format(self, record: logging.LogRecord) -> str:
        time: str = record.asctime
        name: str | None = record.msg
        location: str = record.name
        level: str = record.levelname
        process: int | None = record.process
        thread: int | None = record.thread
        # Additional arguments, will always be a dict via control over the actual log function.
        args: Mapping[str, Any] = record.args # ty: ignore[invalid-assignment] 
        shape: RecordShape = record.type # ty: ignore[unresolved-attribute] Added via extra
        title: str | None = record.title # ty: ignore[unresolved-attribute] Added via extra
        
        element = f'{level}:{location}'

        if name is not None:
            element += f':{name}'

        if thread is not None or process is not None:
            element += '--'
            if process is not None:
                element += str(process)
            if thread is not None:
                element += ':' + str(thread)

        additional = ''
        if title is not None:
            additional += ' :title="' + title.format(name=str(name)) + '"'
        for (key, value) in args.items():
            additional += ' ' + key + '="' + str(value) + '"'
        
        start_tag = '<'
        body = f'{element} :timestamp="{time}"{additional}'
        end_tag = '>'

        if shape is RecordShape.CLOSE:
            start_tag = start_tag + '/'
        if shape is RecordShape.SINGLE:
            end_tag = '/' + end_tag
        
        return f'{start_tag}{body}{end_tag}\n'

class ConsoleFormatter(logging.Formatter):
    """Formats the log record for printing to the terminal.
    
    Overrides the default format method.
    """
    
    def __init__(self):
        super(ConsoleFormatter, self).__init__()

    @override
    def format(self, record: logging.LogRecord) -> str:
        name: str | None = record.msg
        location: str = record.name
        level: str = record.levelname
        process: int | None = record.process
        thread: int | None = record.thread
        # Additional arguments, will always be a dict via control over the actual log function.
        args: Mapping[str, Any] = record.args # ty: ignore[invalid-assignment] 
        title: str | None = record.title # ty: ignore[unresolved-attribute] Added via extra

        if thread is not None or process is not None:
            location += ':'
            if process is not None:
                location += str(process)
            if thread is not None:
                location += ':' + str(thread)

        title = str(title).format( name= str(name) )

        out = f"{level} at {location}: {title}"
        if len(args) > 0:
            out += "Additional info:"
        for (key, value) in args.values():
            out += f"\n\t{key}: {value}"
        return out + "\n"        

class CleverLogger:
    """A less-basic extention to the default python logger."""
    TRACE = logging.DEBUG - 5 # 5
    DEBUG = logging.DEBUG # 10
    INFO = logging.INFO # 20
    WARN = logging.WARN # 30
    ERROR = logging.ERROR # 40
    CRITICAL = logging.FATAL - 5 # 45
    FATAL = logging.FATAL # 50

    major_process: str
    minor_process: str

    def __init__(self, name):
        self.logger = logging.getLogger(name)
        self.logger.setLevel( os.environ.get('LOG_LEVEL', 'INFO').upper() )

        file_log_level = os.environ.get(
            'FILE_LOG_LEVEL',
            os.environ.get( 'LOG_LEVEL', 'INFO' )
        ).upper()

        stream_log_level = os.environ.get(
            'CONSOLE_LOG_LEVEL',
            os.environ.get( 'LOG_LEVEL', 'ERROR' )
        ).upper()

        file_handler = logging.FileHandler('./logs/latest.log.xml')
        file_handler.setLevel( file_log_level )
        file_handler.setFormatter( XMLFormatter() )

        console_handler = logging.StreamHandler()
        console_handler.setLevel( stream_log_level )
        console_handler.setFormatter( ConsoleFormatter() )

        self.logger.addHandler(file_handler)
        self.logger.addHandler(console_handler)


    # TRACE
    def log_open_control_flow(self, control_flow_element: str, **kwargs: Dict[str, Any]):
        self.logger.log(self.TRACE, control_flow_element, kwargs, extra={
            'type': RecordShape.OPEN,
            'title': 'Entering {name}.'
        })

    def log_close_control_flow(self, control_flow_element: str,):
        self.logger.log(self.TRACE, control_flow_element, extra={
            'type': RecordShape.CLOSE,
            'title': 'Leaving {name}.'
        }) 

    def log_control_element(self, control_flow_element: str, **kwargs: Dict[str, Any]):
        self.logger.log(self.TRACE, control_flow_element, kwargs, extra={
            'type': RecordShape.SINGLE,
            'title': '{name}.'
        })

    def log_enter_function(self, function_name: str, **kwargs: Dict[str, Any]):
        self.logger.log(self.TRACE, function_name, kwargs, extra={
            'type': RecordShape.OPEN,
            'title': '{name} called.'
        })

    def log_exit_function(self, function_name: str):
        self.logger.log(self.TRACE, function_name, extra={
            'type': RecordShape.CLOSE,
            'title': '{name} exited.'
        }) 

    def log_function_exit_type(self, exit_type: str, **kwargs: Dict[str, Any]):
        self.logger.log(self.TRACE, exit_type, kwargs, extra={
            'type': RecordShape.SINGLE,
            'title': 'Exited via {name}.'
        })

    # DEBUG
    def log_start_load_module(self, module_name: str, **kwargs: Dict[str, Any]):
        self.logger.log(self.DEBUG, module_name, kwargs, extra={
            'type': RecordShape.OPEN,
            'title': 'Loading module {name}...'
        })

    def log_end_load_module(self, module_name: str,):
        self.logger.log(self.DEBUG, module_name, extra={
            'type': RecordShape.CLOSE,
            'title': 'Loaded module {name}.'
        })

    def log_start_minor_process(self, minor_process_name: str, **kwargs: Dict[str,Any]):
        self.logger.log(self.DEBUG, minor_process_name, kwargs, extra={
            'type': RecordShape.OPEN,
            'title': 'Starting {name}...'
        })

    def log_end_minor_process(self, minor_process: str,):
        self.logger.log(self.DEBUG, minor_process, extra={
            'type': RecordShape.CLOSE,
            'title': 'Finished {name}.'
        })

    def log_variable(self, variable_name: str, **kwargs: Dict[str, Any]):
        self.logger.log(self.DEBUG, variable_name, kwargs, extra={
            'type': RecordShape.SINGLE,
            'title': 'Debug information for {name}.'
        })

    # INFO
    def log_notice(self, notice: str, **kwargs: Dict[str, Any]):
        self.logger.log(self.INFO, None, kwargs, extra={
            'type': RecordShape.SINGLE,
            'title': notice
            })

    def log_start_major_process(self, major_process: str, **kwargs: Dict[str, Any]):
        self.logger.log(self.INFO, major_process, kwargs, extra={
            'type': RecordShape.OPEN,
            'title': 'Starting {name}...'
        })

    def log_end_major_process(self, major_process: str):
        self.logger.log(self.INFO, major_process, extra={
            'type': RecordShape.CLOSE,
            'title': 'Finished {name}.'
        })

    # WARN
    def log_warning(self, warning: str, **kwargs: Dict[str, Any]):
        self.logger.log(self.WARN, None, kwargs, extra={
            'type': RecordShape.SINGLE,
            'title': warning
        })

    # ERROR
    def log_error(self, minor_process: str, error: str, **kwargs: Dict[str, Any]):
        self.logger.log(self.ERROR, minor_process, kwargs, extra={
            'type': RecordShape.SINGLE,
            'title': 'Process {name} encountered an error: ' + error
        })

    # CRITICAL
    def log_critical(self, major_process: str, error: str, **kwargs: Dict[str, Any]):
        self.logger.log(self.ERROR, major_process, kwargs, extra={
            'type': RecordShape.SINGLE,
            'title': 'Process {name} encountered an error: ' + error,
        })

    # FATAL
    def log_fatal(self, deathrattle: str, **kwargs: Dict[str, Any]):
        self.logger.log(self.ERROR, None, kwargs, extra={
            'type': RecordShape.SINGLE,
            'title': deathrattle,
        })
