class NoteNumberOutOfRange(Exception):
    """
    Raised when note number is out of range [0, 127]
    """


class NoSetTempoMessageFound(Exception):
    """
    Raised when no set tempo message found while processing midi file
    """


class NoTimeSignatureMessageFound(Exception):
    """
    Raised when no time signature message found while processing midi file
    """
