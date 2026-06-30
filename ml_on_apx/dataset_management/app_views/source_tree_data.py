from rich.style import Style
from rich.text import Text
from ml_on_apx.labelling import Label
from pathlib import Path


class SourceTreeData(object):
    class NotASourceException(Exception):
        pass

    class NotADirectoryException(Exception):
        pass

    def __init__(self):
        self._is_directory: bool = False
        self.decendant_included: bool = False
        self.decendant_error: bool = False

        self.included = False
        self.ancestor_included: bool = False

        self._path: Path | None = None
        self._label: Label | None = None
        self._name: str

    @classmethod
    def new_leaf_data(
        cls,
        name: str,
        path: Path,
    ) -> "SourceTreeData":
        data = cls()
        data._name = name
        data._path = path
        return data

    @classmethod
    def new_directory_data(
        cls,
        name: str,
    ) -> "SourceTreeData":
        data = cls()
        data._name = name
        data._is_directory = True
        data.decendant_included = False
        data.decendant_error = False
        return data

    def get_text(self, theme: dict[str, str]) -> Text:
        string = self._name
        if self._is_directory:
            string += "/"
        if self.included and self._label is not None:
            string += f" [{self._label}]"
        text = Text(string)

        if self.ancestor_included:
            text.style = Style(color=theme["foreground-muted"])
        elif self.included:
            if self._label is not None:
                text.style = Style(color=theme["text-success"], bold=True)
            else:
                text.style = Style(color=theme["text-error"], bold=True)
        elif self._is_directory:
            if self.decendant_included == 0:
                text.style = Style(color=theme["foreground"])
            elif self.decendant_error == 0:
                text.style = Style(color=theme["text-warning"], bold=True)
            else:
                text.style = Style(color=theme["foreground"], bold=True)
        else:
            text.style = Style(color=theme["foreground"])

        return text

    def is_directory(self) -> bool:
        return self._is_directory

    def get_path(self) -> Path:
        if self._is_directory:
            raise self.NotASourceException()
        assert self._path is not None
        return self._path

    def get_label(self) -> Label | None:
        if self._is_directory:
            raise self.NotASourceException()
        return self._label

    def set_label(self, label: Label | None):
        self._label = label

    def has_error(self) -> bool:
        if self._is_directory:
            assert self.decendant_error is not None
            return self.decendant_error == 0
        return self._label is not None


"""

# Leaf node: ROOT Dataset
States:
- Not included              ($foreground)
- Ancestor included         ($foreground.muted)
- Included with label       ($text-success, bold)
- Included, label not set   ($text-error, bold)
Data:
- Path to source
- Included/not included
- Label, if included
Messages:
- Selected
- LabelSet

# Node: Directory
States:
- No decendant included                         ($foreground)
- Ancestor included                             ($foreground.muted)
- (Some) decendants included without error      ($foreground, bold)
- (Some) decendants included with error         ($text-warning, bold)
- Directly included with label                  ($text-success, bold)
- Directly included, label not set              ($text-error, bold)
Data:
- Path to directory
- Directly included/not directly included
- Label, if directly included
Messages:
- Selected
- LabelSet
Message Handlers:
- leaf.Selected
- leaf.


"""
