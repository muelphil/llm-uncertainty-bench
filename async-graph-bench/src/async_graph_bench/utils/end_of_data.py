class EndOfData:
    """
    Marker class used to signal the end of data stream.

    Upon receiving an instance of EndOfData, nodes and surrounding layers
    should perform cleanup and finalize processing.
    """
    pass
