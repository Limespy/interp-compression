from .. import _lnumba as nb
# ======================================================================
Callable = tuple
Any = T = object
N_CoeffsTV = N_DiffsTV = N_PointsTV = N_SamplesTV = N_VarsTV = object
Length = int
# ----------------------------------------------------------------------
N_Compressed = N_Uncompressed = N_Points = N_Coeffs = N_Diffs = Length
N_Vars = BufferSize = Length
# ----------------------------------------------------------------------
X = nb.A(1)
XSingle = nb.f64
# ----------------------------------------------------------------------
Y = NotImplemented

YLine = nb.A(2)
YDiff = YDiff0 = YDiff1 = YDiff2 = YDiff3 = nb.A(3)

YSingle = NotImplemented

YLineSingle = nb.A(1)
YDiffSingle = nb.A(2)
# ----------------------------------------------------------------------
Excess = nb.f32
Index = nb.up
fIndex = nb.f32
# ----------------------------------------------------------------------
Coeffs = nb.A(2)
TolsDiff = nb.ARO(2)
TolsLine = nb.ARO(1)
