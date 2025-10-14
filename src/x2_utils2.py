

import re


def pid_plus_title( 
        qid, 
        title, 
        sent_idx
        ):
    """

    ##########################################################

    """
    if not title:
        safe = "no_title"
    else:

        safe = re.sub(r"[^0-9A-Za-z]+", "_", title.lower()).strip("_")
        if not safe:
            safe = "no_title"
    return f"{qid}__{safe}_sent{sent_idx}"



