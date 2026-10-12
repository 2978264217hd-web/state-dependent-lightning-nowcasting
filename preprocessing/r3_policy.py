"""R3 input/temporal predicates only. Does not construct flash labels."""
def window_intersects(start,end,invalid_start,invalid_end):
    if end<start or invalid_end<=invalid_start:raise ValueError('Invalid interval')
    return start<invalid_end and end>invalid_start and start<end
def glm_anchor_invalid(t,invalid_intervals):
    """Numeric epoch seconds; half-open 30 min history and future windows."""
    return any(window_intersects(t-1800,t,a,b) or window_intersects(t,t+1800,a,b) for a,b in invalid_intervals)
def primary_candidate(value_raw,qc_class,source_time_space_valid=True):
    """Return -1 unknown, 0 false, 1 true. QC classes: physical0,-99=1,-999=2,decode3."""
    import numpy as np
    value=np.asarray(value_raw);classes=np.asarray(qc_class)
    if value.shape!=classes.shape:raise ValueError('Value/class shape mismatch')
    truth=np.full(value.shape,-1,dtype='i1')
    if source_time_space_valid:
        truth[classes==1]=0
        physical=(classes==0)&np.isfinite(value)
        truth[physical]=(value[physical]>=30).astype('i1')
    return truth
