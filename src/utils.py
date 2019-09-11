import numpy as np
import logging

def reduce_mem_usage(props):
    start_mem_usg = props.memory_usage().sum() / 1024**2
    logging.info(f"Memory usage of properties dataframe is: {start_mem_usg}MB")
    na_list = []
    for col in props.columns:
        if props[col].dtype == object:
            continue
        logging.info(f"Previous type of column {col} props[col].dtype")

        is_int = False
        mx = props[col].max()
        mn = props[col].min()

        # Integer does not support NA, NA needs to be filled
        # TODO: better handling of nans
        if not np.isfinite(props[col]).all():
            na_list.append(col)
            props[col].fillna(mn-1, inplace=True)

        # Test if column can be converted to int
        asint = props[col].fillna(0).astype(np.int64)
        result = np.abs(props[col] - asint)
        result = result.sum()
        if result < 0.01:
            is_int = True

        if is_int:
            if mn-1 >= 0:
                if mx < 255:
                    props[col] = props[col].astype(np.uint8)
                elif mx < 65535:
                    props[col] = props[col].astype(np.uint16)
                elif mx < 4294967295:
                    props[col] = props[col].astype(np.uint32)
                else:
                    props[col] = props[col].astype(np.uint64)
            else:
                if mn-1 > np.iinfo(np.int8).min and mx < np.iinfo(np.int8).max:
                    props[col] = props[col].astype(np.int8)
                elif mn-1 > np.iinfo(np.int16).min and mx < np.iinfo(np.int16).max:
                    props[col] = props[col].astype(np.int16)
                elif mn-1 > np.iinfo(np.int32).min and mx < np.iinfo(np.int32).max:
                    props[col] = props[col].astype(np.int32)
                elif mn-1 > np.iinfo(np.int64).min and mx < np.iinfo(np.int64).max:
                    props[col] = props[col].astype(np.int64)
        else:
            props[col] = props[col].astype(np.float32)

        logging.info(f"New type of column {col} props[col].dtype")
    end_mem_usg = props.memory_usage().sum() / 1024**2
    logging.info(f"Memory usage of properties dataframe after optimisation is: {start_mem_usg}MB")
    logging.info(f"This is {100 * end_mem_usg / start_mem_usg}% if the original size")
    return props
